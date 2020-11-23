# Useful MessageLogger Configuration Changes

## Turning off the end of job statistics
This requires setting the parameter of cerr
```python
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enableStatistics = False
```

## Switching from cerr to cout
One needs to switch off cerr and switch on cout

```python
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True
```

## Writing the standard output to a file
Assing the file you want to write is name `my_file.log` then

```python
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.files.my_file = dict()
```

The `MessageLogger` PSet knows that all parameters of `files` must be `cms.untracked.PSet` so one can use a python `dict` to allow the default values to be used.

## Have specific info messages be displayed
By default the `cerr` will reject all messages below `FWKINFO` (i.e. `threshold = "FWKINFO"`) and the defaut for `INFO` is set to print no output ,i.e. `limit = 0`. In order to see messages for the category 'MyCat' the threshold must be lowered and the category must be explicitly mentioned

```python
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MyCat = dict()
```

## Have debug message show for a given module
By default, all `LogDebug` code is actually removed at compilation time so any messages you want to see have to be recompiled after setting the `EDM_ML_DEBUG` compilation parameter

```bash
> export EDM_ML_DEBUG=1
> scram b ...
```

Then in the `MessageLogger` configuration you need to lower the `threshold` to `"DEBUG"` and then say you want debug messages from the module. So if your module uses the label `myModule` in the configuration, you'd specify

```python
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["myModule"]
```
