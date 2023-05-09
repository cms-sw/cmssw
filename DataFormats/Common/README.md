#  DataFormats/Common

## `edm::TriggerResults`

The class `edm::TriggerResults` is part of the RAW data, and any changes must be backwards compatible. In order to ensure it can be read by all future CMSSW releases, there is a `TestTriggerResultsFormat` unit test, which makes use of the `TestReadTriggerResults` analyzer and the `TestWriteTriggerResults` producer. The unit test checks that the object can be read properly from

* a file in the same release as it was written
* files written by (some) earlier releases can be read

If the persistent format of class `edm::TriggerResults` gets changed in the future, please adjust the `TestReadTriggerResults` and `TestWriteTriggerResults` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to [https://github.com/cms-data/DataFormats-Common/](https://github.com/cms-data/DataFormats-Common/) repository, and update the `TestTriggerResultsFormat` unit test to read the newly created file. The file name should contain the release or pre-release with which it was written.`
