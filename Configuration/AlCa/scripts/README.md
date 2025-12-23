# scripts/

This directory contains small utilities related to Configuration/AlCa used for inspecting and working with autoCond mappings and similar helper tasks.

## Contents
- `printAutoCond.py` — CLI tool to inspect `Configuration.AlCa.autoCond`:
  - list available keys
  - show a specific key's GlobalTag
  - filter keys by substring
  - JSON output for scripting

## Quick start

1. Source a CMSSW environment so `CMSSW_VERSION` is set (e.g. `cmsenv`).
2. Run the utility from this folder (or provide full path):

```bash
printAutoCond.py --list
printAutoCond.py --key run2_design --json
printAutoCond.py --pattern run2
```

Run `python printAutoCond.py --help` for all options.
