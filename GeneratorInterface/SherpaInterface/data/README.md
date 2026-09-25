# SherpaInterface data utilities

## convert_sherpack.sh

Converts a Sherpa sherpack between the **new** (`.tar.xz`) and **old** (`.tgz`) formats,
and generates the matching `_cff.py` fragment from the `Run.dat` inside the sherpack.

### Formats

| Format | Filename pattern | Used with |
|--------|-----------------|-----------|
| new | `sherpa_<PROCESS>_<SCRAM_ARCH>_<CMSSW_VERSION>.tar.xz` | CMSSW ≥ 14 (xz-aware `SherpackFetcher`) |
| old | `sherpa_<PROCESS>.tgz` | legacy gzip-based `SherpackFetcher` |

### Prerequisites

Run inside a CMSSW environment (`cmsenv`) so that `SCRAM_ARCH` and `CMSSW_VERSION`
are set. `tar` and `python3` must be available (both are standard in the Singularity image).

```bash
cd /your/work/dir
cmsenv
```

### new → old

Convert a `.tar.xz` sherpack to a `.tgz` and generate its `_cff.py`:

```bash
convert_sherpack.sh new_to_old sherpa_DY_MASTER_el8_amd64_gcc12_CMSSW_14_0_21.tar.xz
```

Produces:

| File | Description |
|------|-------------|
| `sherpa_DY_MASTER.tgz` | gzip-compressed sherpack |
| `sherpa_DY_MASTER.md5` | full `md5sum` output line, e.g. `c0f81ea6...  sherpa_DY_MASTER.tgz` |
| `sherpa_DY_MASTER_cff.py` | CMS config fragment |

In the generated `_cff.py`:

- `SherpackLocation` is set to the **parent directory** of the `.tgz` (i.e. `$(pwd)/`).
  Update this if you move the `.tgz` to a different location (e.g. `/cvmfs/...`).
- `SherpackChecksum` is the MD5 hash of the `.tgz`.
- `FetchSherpack = True` — the old fetcher copies the file from `SherpackLocation`.

### old → new

Convert a `.tgz` sherpack to a `.tar.xz` and generate its `_cff.py`:

```bash
convert_sherpack.sh old_to_new sherpa_DY_MASTER.tgz
```

Produces:

| File | Description |
|------|-------------|
| `sherpa_DY_MASTER_el8_amd64_gcc12_CMSSW_14_0_21.tar.xz` | xz-compressed sherpack |
| `sherpa_DY_MASTER_cff.py` | CMS config fragment |

In the generated `_cff.py`:

- `SherpackLocation` is set to the **full absolute path** of the `.tar.xz` (i.e. `$(pwd)/sherpa_...tar.xz`).
  Update this if you move the file.
- `SherpackChecksum` is the MD5 hash of the `.tar.xz`.
- `FetchSherpack = True`.

### Generated _cff.py

The `Run = cms.vstring(...)` block is populated from `Run.dat` found inside the sherpack.
The `MPI_Cross_Sections` block is always:

```python
MPI_Cross_Sections = cms.vstring(
    " MPIs in Sherpa, Model = Amisic:",
    " semihard xsec = 39.2965 mb,",
    " non-diffractive xsec = 17.0318 mb with nd factor = 0.3142."
),
```

`SherpaProcess` is derived from the filename by stripping the `sherpa_` prefix and
optional `_MASTER` suffix (e.g. `sherpa_DY_MASTER` → `DY`). Edit it manually if needed.

### After conversion

Always check the generated `_cff.py` and update:

1. `SherpackLocation` — if the sherpack was moved after conversion.
2. `SherpackChecksum` — must match the MD5 of the file at `SherpackLocation`.
3. `SherpaProcess` — if the auto-derived name does not match.

The `_GEN.py` that runs under `cmsRun` typically `import`s the `_cff.py`, so changes
there propagate automatically.
