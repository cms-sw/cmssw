rawDataTask = dict(
)

rawDataTaskPaths = dict(
    EventTypePreCalib  = "RawData/RawDataTask event type BX lt 3490",
    EventTypeCalib     = "RawData/RawDataTask event type BX eq 3490",
    EventTypePostCalib = "RawData/RawDataTask event type BX gt 3490",
    CRC                = "RawData/RawDataTask CRC errors",
    RunNumber          = "RawData/RawDataTask DCC-GT run mismatch",
    Orbit              = "RawData/RawDataTask DCC-GT orbit mismatch",
    TriggerType        = "RawData/RawDataTask DCC-GT trigType mismatch",
    L1ADCC             = "RawData/RawDataTask DCC-GT L1A mismatch",
    L1AFE              = "RawData/RawDataTask FE-DCC L1A mismatch",
    L1ATCC             = "RawData/RawDataTask TCC-DCC L1A mismatch",
    L1ASRP             = "RawData/RawDataTask SRP-DCC L1A mismatch",
    BXDCC              = "RawData/RawDataTask DCC-GT BX mismatch",
    BXFE               = "RawData/RawDataTask FE-DCC BX mismatch",
    BXTCC              = "RawData/RawDataTask TCC-DCC BX mismatch",
    BXSRP              = "RawData/RawDataTask SRP-DCC BX mismatch",
    DesyncByLumi       = "RawData/RawDataTask sync errors by lumi",
    DesyncTotal       = "RawData/RawDataTask sync errors total",
    FEStatus           = "RawData/FEStatus/RawDataTask FE status",
    FEByLumi           = "RawData/RawDataTask FE status errors by lumi",
    FEDEntries         = '%(hlttask)s/FEDEntries',
    FEDFatal           = '%(hlttask)s/FEDFatal'
)

