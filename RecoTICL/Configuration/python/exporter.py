# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Export an assembled pyTICL configuration to a standalone ``_cff.py`` fragment.

The emitted file is a normal CMSSW config fragment -- module definitions at
module scope followed by the ``cms.Task`` hierarchy -- so it can be pulled into
any process with ``process.load(...)`` (useful for HLT menus / sharing).  Module
bodies come from the deterministic :meth:`dumpPython`; tasks are rendered from
the composition recorded by the assembler, leaf-first so every reference
resolves.
"""

_HEADER = "import FWCore.ParameterSet.Config as cms"


def render_cff(assembled):
    """Return the text of a self-contained cff fragment for ``assembled``."""
    blocks = [_HEADER, ""]
    for label in sorted(assembled.modules):
        blocks.append("%s = %s" % (label, assembled.modules[label].dumpPython()))
    blocks.append("")
    for name, children in assembled.task_children.items():
        blocks.append("%s = cms.Task(%s)" % (name, ", ".join(children)))
    return "\n".join(blocks) + "\n"


def to_cff(cfg, path, process_name="TICL", validate=True):
    """Validate (optionally) and write ``cfg`` as a cff fragment to ``path``.

    ``process_name`` is accepted for API symmetry / future full-process dumps;
    a cff fragment itself carries no process.
    """
    assembled = cfg.validate() if validate else cfg.assemble()
    text = render_cff(assembled)
    with open(path, "w") as handle:
        handle.write(text)
    return path
