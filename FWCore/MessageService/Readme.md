# Useful MessageLogger Configuration Changes

Note that `MessageLogger` is part of `process` by default, i.e. explicit loading of `FWCore.MessageService.MessageLogger_cfi` is not necessary.

## Turning off the end of job statistics
This requires setting the parameter of cerr
```python
process.MessageLogger.cerr.enableStatistics = False
```

## Switching from cerr to cout
One needs to switch off cerr and switch on cout

```python
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True
```

## Writing the standard output to a file
Assuming the file you want to write is name `my_file.log` then

```python
process.MessageLogger.files.my_file = dict()
```

The `MessageLogger` PSet knows that all parameters of `files` must be `cms.untracked.PSet` so one can use a python `dict` to allow the default values to be used.

NOTE: The messages are still routed to cerr. If you only want messages to go to the file, also add
```python
process.MessageLogger.cerr.enable = False
```
## Have all  messages from a certain level be displayed
By default the `cerr` will reject all messages below `FWKINFO` (i.e. `threshold = "FWKINFO"`) and the defaut for `INFO` is set to print no output ,i.e. `limit = 0`. So to see all `INFO` messages one would do

```python
process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.INFO.limit = -1
```

All messages above `FWKINFO` by default are set to be displayed, that is they have `limit = -1`. You can explicitly set that by doing
```python
process.MessageLogger.cerr.WARNING = dict(limit = -1)
```

The `MessageLogger.cerr` PSet knows that all _extra_ parameter labels must be of type `cms.untracked.PSet` so one can use a python `dict` set specific parameters for that PSet and allow all other parameters to use their default default values.


## Have specific info messages be displayed
By default the `cerr` will reject all messages below `FWKINFO` (i.e. `threshold = "FWKINFO"`) and the defaut for `INFO` is set to print no output ,i.e. `limit = 0`. In order to see messages for the category 'MyCat' the threshold must be lowered and the category must be explicitly mentioned

```python
process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MyCat = dict()
```

The `MessageLogger.cerr` PSet knows that all _extra_ parameter labels must be of type `cms.untracked.PSet` so one can use a python `dict` set specific parameters for that PSet and allow all other parameters to use their default default values.


## Suppressing a particular category
In order to supporess messages for a given category, e.g. 'MyCat', the limit should be set to 0

```python
process.MessageLogger.MyCat = dict(limit = 0)
```

The `MessageLogger` PSet knows that all _extra_ parameter labels must be of type `cms.untracked.PSet` so one can use a python `dict` set specific parameters for that PSet and allow all other parameters to use their default default values.


## Have all debug messages be shown

By default, all `LogDebug` code is actually removed at compilation time so any messages you want to see have to be recompiled after setting the `EDM_ML_DEBUG` compilation parameter

```bash
> export USER_CXXFLAGS="-DEDM_ML_DEBUG"
> scram b ...
```

Then in the `MessageLogger` configuration you need to lower the `threshold` to `"DEBUG"`
```python
process.MessageLogger.cerr.threshold = "DEBUG"
```

The default `MessageLogger` configuration leads to all debug messages to be printed (also outside modules).


## Have debug message show for a given module

It is possible to restrict the shown debug messages to specific module(s). For example, if your module uses the label `myModule` in the configuration, you'd specify
```python
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["myModule"]
```

For backwards compatibility, setting `debugModules` to `*` has the same effect as setting `debugModules` empty, i.e. showing all debug messages.
```python
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["*"]
```

