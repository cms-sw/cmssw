# DataFormats/L1Scouting

## L1 Trigger Scouting data formats

Any changes to the L1 scouting data formats must be backwards compatible.
In order to ensure the L1 Scouting formats can be read by future CMSSW releases,
there is a `TestWriteL1ScoutingDataFormats` unit test, which makes use of the `TestReadL1Scouting` analyzer and the `TestWriteL1Scouting` producer.
The unit test checks that objects can be written and read properly. 