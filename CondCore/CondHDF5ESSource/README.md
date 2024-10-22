# Overview of CondCore/CondHDF5ESSource

## CondHDF5ESSource

CondHDF5ESSource is an EventSetup source that can read conditions data from an HDF5 file. The full configuration options for the module can be obtained by doing the standard

```
> edmPluginHelp -p CondHDF5ESSource
```

In general, one will specify the file to read and the global tag stored within that file to use. E.g.

```python

globalTag = cms.ESSource("CondHDF5ESSource", filename = cms.string("JobConditions.h5cond"), globalTag = cms.string("SomeTag") ) 
```

## Utilities for creating an HDF5 conditions file

### conddb2hdf5.py

This utility reads from the conditions database and writes a specified global tag into to the file. The full range of options can be seen by doing

```
> conddb2hdf5.py --help
```

The simplest way to use the utlity is to specify a global tag and the output file name

```
> conddb2hdf5.py SomeTag --output JobConditions.h5cond
```

NOTE: it can take many hours to dump all the conditions from a global tag stored in the DB.

### condhdf5tohdf5.py

This utility reads from one hdf5 conditions file and writes a specified global tag into a new hdf5 file. The full range of options can be seen by doing

```
> condhdf5tohdf5.py --help
```

The simplest way to use the utlity is to specify the input file, a global tag and the output file name

```
> condhdf5tohdf5.py OriginaFile.h5cond SomeTag --output JobConditions.h5cond
```

