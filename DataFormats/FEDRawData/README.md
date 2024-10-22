#  DataFormats/FEDRawData

## `FEDRawDataCollection`

The class `FEDRawDataCollection` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestFEDRawDataCollectionFormat` unit test, which makes use of the `TestReadFEDRawDataCollection` analyzer and the `TestWriteFEDRawDataCollection` producer. The unit test checks that the object can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of class `FEDRawDataCollection` gets changed in the future, please adjust the `TestReadFEDRawDataCollection` and `TestWriteFEDRawDataCollection` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-FEDRawData/](https://github.com/cms-data/DataFormats-FEDRawData/) repository, and update the `TestFEDRawDataCollectionFormat` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.
