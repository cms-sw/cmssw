#  DataFormats/SiStripCluster

## `SiStripApproximateClusterCollection`

The class `SiStripApproximateClusterCollection` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestSiStripApproximateClusterCollection` unit test, which makes use of the `TestReadSiStripApproximateClusterCollection` analyzer and the `TestWriteSiStripApproximateClusterCollection` producer. The unit test checks that the object can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of class `SiStripApproximateClusterCollection` gets changed in the future, please adjust the `TestReadSiStripApproximateClusterCollection` and `TestWriteSiStripApproximateClusterCollection` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-SiStripCluster/](https://github.com/cms-data/DataFormats-SiStripCluster/) repository, and update the `TestSiStripApproximateClusterCollection` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.
