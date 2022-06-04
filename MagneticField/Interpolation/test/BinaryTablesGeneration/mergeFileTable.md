# mergeFileTable Usage

## Usage

```
  mergeFileTable <directory> <base>
```

The two arguments to the program are
1. `<directory>` the path of the directory holding the `s*` subdirectories that define the table set.
2. `<base>` the base name of the the `.bin` and `.index` files to be generated. In general, this should always be `merged` in order to properly be read in cmsRun.

## Action

The program reads all the `*.bin` files in all the subdirectories and concatenates them into one `<base>.bin` file. It also generates a `<base>.index` file which contains the offsets in the newly generated `<base>.bin` file for each `*.bin` file that was concatenated.

The use of the merged file during cmsRun processing rather than reading the individual files provides a factor of 5 speedup in the initialization of the magnetic field. 

