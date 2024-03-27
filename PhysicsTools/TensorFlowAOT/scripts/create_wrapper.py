# coding: utf-8
# flake8: noqa

"""
Script that parses header files created by the AOT compilation and creates another header file
containing a wrapper class (inheriting from tfaot::Wrapper) for models with different batch sizes.
"""

from __future__ import annotations

import os
import re
from collections import namedtuple


HeaderData = namedtuple("HeaderData", [
    "batch_size",
    "prefix",
    "namespace",
    "class_name",
    "n_args",
    "arg_counts",
    "arg_counts_no_batch",
    "n_res",
    "res_counts",
    "res_counts_no_batch",
])

common_header_data = [
    "prefix",
    "namespace",
    "class_name",
    "n_args",
    "n_res",
    "arg_counts_no_batch",
    "res_counts_no_batch",
]


def create_wrapper(
    header_files: list[str],
    model_path: str,
    subsystem: str,
    package: str,
    output_path: str | None = None,
    template: str = "$CMSSW_BASE/src/PhysicsTools/TensorFlowAOT/templates/wrapper.h.in",
) -> None:
    # read header data
    header_data = {}
    for path in header_files:
        data = parse_header(path)
        header_data[data.batch_size] = data

    # sorted batch sizes
    batch_sizes = sorted(data.batch_size for data in header_data.values())

    # set common variables
    variables = {
        "cmssw_version": os.environ["CMSSW_VERSION"],
        "scram_arch": os.environ["SCRAM_ARCH"],
        "model_path": model_path,
        "batch_sizes": batch_sizes,
        "subsystem": subsystem,
        "package": package,
    }
    for key in common_header_data:
        values = set(getattr(d, key) for d in header_data.values())
        if len(values) > 1:
            raise ValueError(f"found more than one possible {key} values: {', '.join(values)}")
        variables[key] = values.pop()

    # helper for variable replacement
    def substituter(variables):
        # insert upper-case variants of strings, csv variants of lists
        variables_ = {}
        for key, value in variables.items():
            key = key.upper()
            variables_[key] = str(value)
            if isinstance(value, str) and not key.endswith("_UC"):
                variables_[f"{key}_UC"] = value.upper()
            elif isinstance(value, (list, tuple)) and not key.endswith("_CSV"):
                variables_[f"{key}_CSV"] = ", ".join(map(str, value))

        def repl(m):
            key = m.group(1)
            if key not in variables_:
                raise KeyError(f"template contains unknown variable {key}")
            return variables_[key]

        return lambda line: re.sub(r"\$\{([A-Z0-9_]+)\}", repl, line)

    # substituter for common variables and per-model variables
    common_sub = substituter(variables)
    model_subs = {
        batch_size : substituter({
            **variables,
            **dict(zip(HeaderData._fields, header_data[batch_size])),
        })
        for batch_size in batch_sizes
    }

    # read template lines
    template = os.path.expandvars(os.path.expanduser(str(template)))
    with open(template, "r") as f:
        input_lines = [line.rstrip() for line in f.readlines()]

    # go through lines and define new ones
    output_lines = []
    while input_lines:
        line = input_lines.pop(0)

        # loop statement?
        m = re.match(r"^\/\/\s+foreach=([^\s]+)\s+lines=(\d+)$", line.strip())
        if m:
            loop = m.group(1)
            n_lines = int(m.group(2))

            if loop == "MODEL":
                # repeat the next n lines for each batch size and replace model variables
                batch_lines, input_lines = input_lines[:n_lines], input_lines[n_lines:]
                for batch_size in batch_sizes:
                    for line in batch_lines:
                        output_lines.append(model_subs[batch_size](line))
            else:
                raise ValueError(f"unknown loop target '{loop}'")

            continue

        # just make common substitutions
        output_lines.append(common_sub(line))

    # prepare the output
    if not output_path:
        output_path = f"$CMSSW_BASE/src/{subsystem}/{package}/tfaot_dev/{variables['prefix']}.h"
    output_path = os.path.expandvars(os.path.expanduser(str(output_path)))
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write lines
    with open(output_path, "w") as f:
        f.writelines("\n".join(map(str, output_lines)) + "\n")


def parse_header(path: str) -> HeaderData:
    # read all non-empty lines
    path = os.path.expandvars(os.path.expanduser(str(path)))
    with open(path, "r") as f:
        lines = [line for line in (line.strip() for line in f.readlines()) if line]

    # prepare HeaderData
    data = HeaderData(*([None] * len(HeaderData._fields)))

    # helper to set data fields
    set_ = lambda key, value: data._replace(**{key: value})

    # extract data
    arg_counts = {}
    res_counts = {}
    while lines:
        line = lines.pop(0)

        # read the namespace
        m = re.match(r"^namespace\s+([^\s]+)\s*\{$", line)
        if m:
            data = set_("namespace", m.group(1))
            continue

        # read the class name and batch size
        m = re.match(rf"^class\s+([^\s]+)_bs(\d+)\s+final\s+\:\s+public\stensorflow\:\:XlaCompiledCpuFunction\s+.*$", line)  # noqa
        if m:
            data = set_("class_name", m.group(1))
            data = set_("batch_size", int(m.group(2)))

        # read argument and result counts
        m = re.match(r"^int\s+(arg|result)(\d+)_count\(\).+$", line)
        if m:
            # get kind and index
            kind = m.group(1)
            index = int(m.group(2))

            # parse the next line
            m = re.match(r"^return\s+(\d+)\s*\;.*$", lines.pop(0))
            if not m:
                raise Exception(f"corrupted header file {path}")
            count = int(m.group(1))

            # store the count
            (arg_counts if kind == "arg" else res_counts)[index] = count
            continue

    # helper to flatten counts to lists
    def flatten(counts: dict[int, int], name: str) -> list[int]:
        if set(counts) != set(range(len(counts))):
            raise ValueError(
                f"non-contiguous indices in {name} counts: {', '.join(map(str, counts))}",
            )
        return [counts[index] for index in sorted(counts)]


    # helper to enforce integer division by batch size
    def no_batch(count: int, index: int, name: str) -> int:
        if count % data.batch_size != 0:
            raise ValueError(
                f"{name} count of {count} at index {index} is not dividable by batch size "
                f"{data.batch_size}",
            )
        return count // data.batch_size

    # store the prefix
    base = os.path.basename(path)
    postfix = f"_bs{data.batch_size}.h"
    if not base.endswith(postfix):
        raise ValueError(f"header '{path}' does not end with expected postfix '{postfix}'")
    data = set_("prefix", base[:-len(postfix)])

    # set counts
    data = set_("n_args", len(arg_counts))
    data = set_("n_res", len(res_counts))
    data = set_("arg_counts", flatten(arg_counts, "argument"))
    data = set_("res_counts", flatten(res_counts, "result"))
    data = set_("arg_counts_no_batch", tuple(
        no_batch(c, i, "argument")
        for i, c in enumerate(data.arg_counts)
    ))
    data = set_("res_counts_no_batch", tuple(
        no_batch(c, i, "result")
        for i, c in enumerate(data.res_counts)
    ))

    return data


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
        help="path of the initial model file for provenance purposes",
    )
    parser.add_argument(
        "--header-files",
        "-f",
        required=True,
        nargs="+",
        help="comma-separated list of AOT header files that define the models to wrap",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        help="path where the created header file should be saved; default: "
        "$CMSSW_BASE/src/SUBSYSTEM/PACKAGE/tfaot_dev/PREFIX.h"
    )
    args = parser.parse_args()

    create_wrapper(
        header_files=args.header_files,
        model_path=args.model_path,
        subsystem=args.subsystem,
        package=args.package,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
