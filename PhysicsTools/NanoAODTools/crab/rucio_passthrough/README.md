# NanoAOD pass-through with CRAB and Rucio

This setup submits NanoAOD files to CRAB and writes every input event and branch
to a new output file without applying a selection or running an analysis module.
Its purpose is to let CRAB stage the resulting files out through Rucio, including
NanoAOD-derived ROOT files that cannot be opened by `cmsRun` because they do not
contain the complete EDM metadata.

See the CRAB documentation for the general
[Rucio stage-out model](https://cmscrab-userguide.docs.cern.ch/rucio-stageout/rucio-stageout.html)
and the [Rucio stage-out tutorial](https://cmscrab-userguide.docs.cern.ch/rucio-stageout/tutorial.html).

## How it works

- `PSet.py` is a dummy CMSSW configuration used by CRAB to discover the expected
  `tree.root` output. It is not the processing payload.
- `crab_script.sh` installs the shipped CMSSW Python area in the worker-node
  environment and starts the payload with Python 3.
- `crab_script.py` runs the NanoAODTools `PostProcessor` with no cut and no
  modules. This selects its full-clone path, preserves the run/luminosity trees,
  and writes a framework job report for CRAB.
- With `fwkJobReport=True`, NanoAODTools merges the per-input clones into the
  final `tree.root` using `haddnano.py`.

The output is a logical clone of the NanoAOD content, not a byte-for-byte copy of
the original ROOT file.

## Configure and submit

Start from `crab_cfg.py` and update at least:

- `General.requestName`
- `Data.inputDataset` and, if needed, `Data.inputDBS`
- the job splitting and units
- `Data.outputDatasetTag`
- `Site.storageSite`

The Rucio destination is enabled by the `/rucio/` component in
`Data.outLFNDirBase`:

```python
config.Data.outLFNDirBase = '/store/user/rucio/%s/NanoPost' % (
    getUsernameFromSiteDB())
```

The user needs sufficient Rucio quota at `Site.storageSite`. Keep
`Data.publication = False` for NanoAOD-derived files without complete EDM
metadata; CRAB still creates Rucio containers for the staged output.

After setting up the CRAB client and proxy, submit from this directory:

```console
crab submit -c crab_cfg.py
```

Use `crab status` to follow both job execution and the asynchronous Rucio
transfer. Once transfers begin, it reports the transfer container and its Rucio
rule.

## Important details

- The payload must use `python3`. In current CMSSW EL8 environments, bare
  `python` is the system Python 2 interpreter and cannot load NanoAODTools.
- Do not set `maxEntries=-1`. NanoAODTools does not use the `cmsRun` convention
  where `-1` means all events. Omit `maxEntries` to process the complete input.
- `cmsRun` is deliberately not used. A `PoolSource` requires EDM `MetaData` and
  `ParameterSets` trees, which may be absent from postprocessed NanoAOD files.
- `modules=[]` is sufficient for pass-through operation; a custom module whose
  `analyze` method only returns `True` is unnecessary.
