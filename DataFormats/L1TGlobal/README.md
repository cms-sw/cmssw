#  DataFormats/L1TGlobal

## `GlobalObjectMapRecord`

The class `GlobalObjectMapRecord` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestGlobalObjectMapRecordFormat` unit test, which makes use of the `TestReadGlobalObjectMapRecord` analyzer and the `TestWriteGlobalObjectMapRecord` producer. The unit test checks that the object can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of class `GlobalObjectMapRecord` gets changed in the future, please adjust the `TestReadGlobalObjectMapRecord` and `TestWriteGlobalObjectMapRecord` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-L1TGlobal/](https://github.com/cms-data/DataFormats-L1TGlobal/) repository, and update the `TestGlobalObjectMapRecord` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.
