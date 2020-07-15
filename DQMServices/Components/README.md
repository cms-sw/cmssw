DQM Services: Components
========================

These packages contain functionality for DQM that is not specific to any data, subsystem or detector. This is the "DQM Framework", provided by the DQM group to allow detector specific code (mostly in `DQM/`, `DQMOffline/` and `Validation/`) to interact with the DQM infrastructure (DQM GUI, Tier0 processing, Online DQM, etc.).

Package Contents
----------------

- `Components/`: Collection of (independent) DQM plugins that handle core functionality. This includes
    - `QualityTester`: Applies Quality Tests defined in XML configuration to MEs
    - `DQMDaqInfo`: TODO: ?
    - `DQMEventInfo`: An analyzer that collects meta data into DQM `MonitorElement`s.
    - `DQMHarvestingMetadata`: A havresting module that collects similar data to `DQMEventInfo`.
    - `DQMFEDIntegrityClient`: TODO: ?
    - `DQMMessageLogger`: Creates histograms of number of EDM error messages.
    - `DQMMessageLoggerClient`: Some sort of post-processing on these histograms.
    - `DQMFileSaver`: Triggers legacy format saving of DQM output in harvesting jobs. See also: `DQMServices/FileIO`.
    - `DQMLumiMonitor`: Some sort of SiPixel based luminosity measurement. Does not belong into this package. Used online only.
    - `DQMScalInfo`: Reports some data from Lumi/L1T Scalers. Used online.
    - `DQMProvInfo`: Populates the `Info/*Info` histograms.
    - `DQMStoreStats`: Provides some DQM self-monitoring (memory etc.) Not used in production.
    - `EDMtoMEConverter`: Reads histograms from EDM event files and puts them into DQM. Used for AlCa. See also `DQMServices/FwkIO`.
    - `MEtoEDMConverter`: Reads histograms from the DQMStore and saves them into EDM products. Used for AlCa.
    - `scripts/`: Tools to acces DQMGUI and to compare/inspect DQM data files.
    - `test/`: DQM unit tests.
