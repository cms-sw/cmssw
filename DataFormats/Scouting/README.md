#  DataFormats/Scouting

## Scouting Data Formats

Any changes to the Scouting data formats must be backwards compatible. In order to ensure the Scouting formats can be read by all future CMSSW releases, there is a `TestRun3ScoutingDataFormats` unit test, which makes use of the `TestReadRun3Scouting` analyzer and the `TestWriteRun3Scouting` producer. The unit test checks that the objects can be read properly from

* a file written by the same release
* files written by (some) earlier releases

If the persistent format of any Scouting data format gets changed in the future, please adjust the `TestReadRun3Scouting` and `TestWriteRun3Scouting` modules accordingly. It is important that every member container has some content in this test. Please also add a new file to the [https://github.com/cms-data/DataFormats-Scouting/](https://github.com/cms-data/DataFormats-Scouting/) repository, and update the `TestRun3ScoutingDataFormats` unit test to read the newly created file. The file name should contain the version numbers of the data format classes (from classes_def.xml) in alphabetical order (they are in this alphabetical order already in classes_def.xml) and the release or pre-release with which it was written. If the latest file of Run 3 scouting before the update has not been used in data taking, the file can be deleted.

There are analogous tests for Run 2. It is unlikely those formats will change anymore. There will probably be analogous tests added in the future for runs after Run 3 which will need similar maintenance.
