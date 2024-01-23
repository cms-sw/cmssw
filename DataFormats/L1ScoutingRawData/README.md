# DataFormats/L1ScoutingRawData

## L1 Trigger Scouting raw data formats

Any changes to the L1 scouting raw data `SDSRawDataCollection` must be backwards compatible.
In order to ensure the L1 Scouting raw data formats can be read by future CMSSW releases,
there is a `TestSDSRawDataCollectionFormat` unit test, which makes use of the `TestReadSDSRawDataCollection` analyzer and the `TestWriteSDSRawDataCollection` producer.
The unit test checks that objects can be written and read properly.