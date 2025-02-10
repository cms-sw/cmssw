# Pixel Barycentre Validation
Currently only the `single` type of job is implemented in the All-in-One Tool, but the execution of the plotting script `plotBaryCentre_VS_BeamSpot.py` could be added in the future as a `trend` job.

## Extraction from TrackerAlignmentRcd - PixelBarycentreAnalyzer_cfg.py
This config runs `PixelBaryCentreAnalyzer` on an empty source, after configuring the GlobalTag and additional conditions for a given alignment.
The following parameters are expected/accepted in the json/yaml configuration file:
```
validations:
    PixBary:
        single:
            <job_name>:
                <options>
```

The following options are understood:

Variable | Default value | Explanation/Options
-------- | ------------- | --------------------
firstRun | 290550 | The first run to process (inclusive)
lastRun | 325175 | The last run to process (inclusive)
lumisPerRun | 1 | The number of LumiSections tested for a change in the TrackerAlignmentRcd in each run
alignments | None | List of alignments for which this validation is run
