# coding: utf-8

"""
Script to simplify the development workflow for compiling and integrating TF AOT models into CMSSW.
"""

from __future__ import annotations

import os
import shutil
import tempfile


tool_name_template = "tfaot-dev-{subsystem}-{package}-{model_name}"

ld_flag_template = "<flags LDFLAGS=\"$LIBDIR/{model_name}_bs{bs}.o\"/>"

tool_file_template = """\
<tool name="{tool_name}" version="{model_version}">
  <client>
    <environment name="TFAOT_DEV_{subsystem_uc}_{package_uc}_{model_name_uc}" default="$CMSSW_BASE/src/{subsystem}/{package}/tfaot_dev"/>
    <environment name="LIBDIR" default="$TFAOT_DEV_{subsystem_uc}_{package_uc}_{model_name_uc}/{lib_dir_name}"/>
    <environment name="INCLUDE" default="$TFAOT_DEV_{subsystem_uc}_{package_uc}_{model_name_uc}/{inc_dir_name}"/>
  </client>
  <use name="tensorflow-xla-runtime"/>
  {ld_flags}
</tool>
"""  # noqa


def compile_model(
    subsystem: str,
    package: str,
    model_path: str,
    model_name: str | None = None,
    model_version: str = "1.0.0",
    batch_sizes: tuple[int] = (1,),
    output_path: str | None = None,
):
    # default output path
    if not output_path:
        # check that we are located in a cmssw env
        cmssw_base = os.getenv("CMSSW_BASE")
        if not cmssw_base or not os.path.isdir(cmssw_base):
            raise Exception("CMSSW_BASE is not set or points to a non-existing directory")

        output_path = os.path.join("$CMSSW_BASE", "src", subsystem, package, "tfaot_dev")
    output_path = os.path.expandvars(os.path.expanduser(output_path))

    # check that the model exists
    model_path = os.path.expandvars(os.path.expanduser(model_path))
    model_path = os.path.normpath(os.path.abspath(model_path))
    if not os.path.exists(model_path):
        raise Exception(f"model_path '{model_path}' does not exist")

    # infer the model name when none was provided
    if not model_name:
        model_name = os.path.splitext(os.path.basename(model_path))[0]

    # prepare directories
    lib_dir = os.path.join(output_path, "lib")
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
    inc_dir = os.path.join(output_path, "include")
    if not os.path.exists(inc_dir):
        os.makedirs(inc_dir)

    # compile the model into a temporary directory
    from cmsml.scripts.compile_tf_graph import compile_tf_graph
    with tempfile.TemporaryDirectory() as tmp_dir:
        compile_tf_graph(
            model_path=model_path,
            output_path=tmp_dir,
            batch_sizes=batch_sizes,
            compile_prefix=f"{model_name}_bs{{}}",
            compile_class=f"{subsystem}_{package}::{model_name}_bs{{}}",
        )

        # copy files
        header_files = []
        for bs in batch_sizes:
            header_name = f"{model_name}_bs{bs}.h"
            shutil.copy2(os.path.join(tmp_dir, "aot", header_name), inc_dir)
            shutil.copy2(os.path.join(tmp_dir, "aot", f"{model_name}_bs{bs}.o"), lib_dir)
            header_files.append(os.path.join(inc_dir, header_name))

    # create the wrapper header
    from create_wrapper import create_wrapper
    create_wrapper(
        header_files=header_files,
        model_path=model_path,
        subsystem=subsystem,
        package=package,
        output_path=os.path.join(inc_dir, f"{model_name}.h"),
    )

    # create the toolfile
    tool_vars = {
        "subsystem": subsystem,
        "subsystem_uc": subsystem.upper(),
        "package": package,
        "package_uc": package.upper(),
        "model_name": model_name,
        "model_name_uc": model_name.upper(),
        "model_version": model_version,
        "lib_dir_name": os.path.basename(lib_dir),
        "inc_dir_name": os.path.basename(inc_dir),
        "tool_name": tool_name_template.format(
            subsystem=subsystem.lower(),
            package=package.lower(),
            model_name=model_name.lower(),
        ),
        "ld_flags": "\n  ".join([
            ld_flag_template.format(model_name=model_name, bs=bs)
            for bs in batch_sizes
        ]),
    }
    tool_path = os.path.join(output_path, f"{tool_vars['tool_name']}.xml")
    with open(tool_path, "w") as f:
        f.write(tool_file_template.format(**tool_vars))

    # print a message
    tool_path_repr = os.path.relpath(tool_path)
    if tool_path_repr.startswith(".."):
        tool_path_repr = tool_path
    inc_path = f"{output_path}/include/{model_name}.h"
    if "CMSSW_BASE" in os.environ and os.path.exists(os.environ["CMSSW_BASE"]):
        inc_path_rel = os.path.relpath(inc_path, os.path.join(os.environ["CMSSW_BASE"], "src"))
        if not inc_path_rel.startswith(".."):
            inc_path = inc_path_rel

    print("\n" + 80 * "-" + "\n")
    print(f"created custom tool file for AOT compiled model '{model_name}'")
    print("to register it to scram, run")
    print(f"\n> scram setup {tool_path_repr}\n")
    print("and use the following to include it in your code")
    print(f"\n#include \"{inc_path}\"\n")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=__doc__.strip(),
    )
    parser.add_argument(
        "--subsystem",
        "-s",
        required=True,
        help="the CMSSW subsystem that the plugin belongs to",
    )
    parser.add_argument(
        "--package",
        "-p",
        required=True,
        help="the CMSSW package that the plugin belongs to",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        help="the path to the model to compile",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="a custom model name; when empty, a name is inferred from --model-path",
    )
    parser.add_argument(
        "--model-version",
        default="1.0.0",
        help="a custom model version; default: 1.0.0",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        default=(1,),
        type=(lambda s: tuple(map(int, s.strip().split(",")))),
        help="comma-separated list of batch sizes to compile the model for; default: 1",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        help="path where the outputs should be saved; default: "
        "$CMSSW_BASE/src/SUBSYSTEM/PACKAGE/tfaot_dev",
    )
    args = parser.parse_args()

    compile_model(
        subsystem=args.subsystem,
        package=args.package,
        model_path=args.model_path,
        model_name=args.model_name,
        model_version=args.model_version,
        batch_sizes=args.batch_sizes,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
