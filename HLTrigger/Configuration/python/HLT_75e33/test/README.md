# Tools for the Phase-2 HLT menu

The Run-3 menus are generated from ConfDB, so their python files are consistent by
construction and come with `getHLT.sh`, `hltListPaths`, `getEventContent.py` and
`getDatasets.py`. The Phase-2 menu in `HLT_75e33/` is maintained by hand, one `_cfi`
per object. These tools fill the same roles for it.

## hltPhase2MenuCheck: static checks of the tree

Parses every file of `HLT_75e33/` (nothing is executed, no `cmsenv` needed) and reports:

| Code | Severity | Meaning |
|------|----------|---------|
| E001 | error    | syntax error |
| E002 | error    | `fragment.load(...)` or import of a file that does not exist |
| E003 | error    | symbol used but not defined or imported in that file |
| W101 | warning  | file not reached from any entry point (orphan) |
| W102 | warning  | file imported but nothing from it is used (stale import) |
| W103 | warning  | `X_cfi.py` does not define `X` |
| W104 | warning  | same symbol defined in several files |
| W105 | warning  | same file loaded twice by an entry point |

```bash
hltPhase2MenuCheck                    # errors make it exit 1
hltPhase2MenuCheck --strict           # warnings too
hltPhase2MenuCheck --ignore W102 --json findings.json
```

Entry points are the files in `HLTrigger/Configuration/python` that mention `HLT_75e33`
(for example `HLT_75e33_cff.py`, `HLT_NGTScouting_cff.py`); add others with `--root FILE`.
E003 is skipped for files whose star imports reach outside the tree, since those
symbols cannot be resolved statically.

## hltPhase2MenuDump: listings, JSON, HTML menu, menu diff

Works on any configuration that contains the menu, for example:

```bash
cmsDriver.py Phase2 -s L1P2GT,HLT:75e33 --processName=HLTX \
  --conditions auto:phase2_realistic_T33 --geometry ExtendedRun4D110 --era Phase2C17I13M9 \
  --eventcontent FEVTDEBUGHLT --filein=file:step1.root --mc -n 10 --no_exec
```

```bash
hltPhase2MenuDump Phase2_L1P2GT_HLT.py --list paths         # cf. hltListPaths
hltPhase2MenuDump Phase2_L1P2GT_HLT.py --list seeds         # path -> L1 seed expression
hltPhase2MenuDump Phase2_L1P2GT_HLT.py --list datasets      # cf. getDatasets.py
hltPhase2MenuDump Phase2_L1P2GT_HLT.py --list eventcontent  # cf. getEventContent.py

# HTML site + JSON model (drop the L1 GT paths with --hlt-only)
hltPhase2MenuDump Phase2_L1P2GT_HLT.py --hlt-only -o html_default --json default.json

# another variant, with a diff page against the default menu
hltPhase2MenuDump Phase2_Alpaka_L1P2GT_HLT.py --hlt-only -o html_alpaka --compare default.json
```

Arguments after a lone `--` are passed to the configuration. A JSON dump can be used as
input instead of a configuration, which avoids reloading the menu.

The HTML site is static and self-contained (no network access, no external assets).
It has:

- a sortable, searchable path table with L1 seed, module and filter counts and the last filter;
- one page per path showing the module chain in order, with filters, Task members,
  negated/ignored modules, the selection parameters of each filter, and products the
  path consumes without scheduling their producers;
- one page per module with plugin type, a link to the plugin source and the `_cfi` file,
  the paths using it, its inputs and consumers, and the full python dump;
- EventSetup, services, outputs and datasets pages;
- a checks page: modules in no path, unscheduled paths, paths without an L1 seed module;
- with `--compare`, a differences page (paths and modules added, removed, reordered or reconfigured).

## Unit tests

`HLTrigger/Configuration/test`:

- `test_hltPhase2MenuCheck.py`: self-test of the checker on a synthetic tree with planted defects;
- `test_hltPhase2MenuTools.sh`: static checks, cmsDriver (`--no_exec`) for the default and alpaka
  menus, dumps, model sanity checks and the diff. Conditions, geometry and era can be overridden
  with `P2_CONDITIONS`, `P2_GEOMETRY` and `P2_ERA`.

```bash
scram b runtests_testHLTPhase2MenuTools
```

## Timing

`runHLTTiming.sh` (this directory) runs the Phase-2 HLT timing benchmark used by the
`hlt_p2_timing` bot test.
