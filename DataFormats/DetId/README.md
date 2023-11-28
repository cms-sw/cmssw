#  DataFormats/DetId

## `std::vector<DetId>`

The type `std::vector<DetId>` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestVectorDetId` unit test, which makes use of the `TestReadVectorDetId` analyzer and the `TestWriteVectorDetId` producer. The unit test checks that the object can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of class `std::vector<DetId>` gets changed in the future, please adjust the `TestReadVectorDetId` and `TestWriteVectorDetId` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-DetId/](https://github.com/cms-data/DataFormats-DetId/) repository, and update the `TestVectorDetId` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.
