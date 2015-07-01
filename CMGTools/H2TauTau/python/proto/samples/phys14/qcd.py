import CMGTools.RootTools.fwlite.Config as cfg

QCDEM10to20 = cfg.MCComponent(
    name          = 'QCDEM10to20',
    files         = []           ,
    xSection      = 1.           ,
    nGenEvents    = 1            ,
    triggers      = []           ,
    effCorrFactor = 1
    )

QCDEM20to30 = cfg.MCComponent(
    name          = 'QCDEM20to30',
    files         = []           ,
    xSection      = 1.           ,
    nGenEvents    = 1            ,
    triggers      = []           ,
    effCorrFactor = 1
    )

QCDEM30to80 = cfg.MCComponent(
    name          = 'QCDEM30to80',
    files         = []           ,
    xSection      = 1.           ,
    nGenEvents    = 1            ,
    triggers      = []           ,
    effCorrFactor = 1
    )

QCDEM80to170 = cfg.MCComponent(
    name          = 'QCDEM80to170',
    files         = []            ,
    xSection      = 1.            ,
    nGenEvents    = 1             ,
    triggers      = []            ,
    effCorrFactor = 1
    )

QCDMu30to50 = cfg.MCComponent(
    name          = 'QCDMu30to50',
    files         = []           ,
    xSection      = 1.           ,
    nGenEvents    = 1            ,
    triggers      = []           ,
    effCorrFactor = 1
    )

QCDMu50to80 = cfg.MCComponent(
    name          = 'QCDMu50to80',
    files         = []           ,
    xSection      = 1.           ,
    nGenEvents    = 1            ,
    triggers      = []           ,
    effCorrFactor = 1
    )

QCDMu80to120 = cfg.MCComponent(
    name          = 'QCDMu80to120',
    files         = []            ,
    xSection      = 1.            ,
    nGenEvents    = 1             ,
    triggers      = []            ,
    effCorrFactor = 1
    )

mc_qcd_em = [
    QCDEM10to20 ,
    QCDEM20to30 ,
    QCDEM30to80 ,
    QCDEM80to170,
    ]

mc_qcd_mu = [
    QCDMu30to50 ,
    QCDMu50to80 ,
    QCDMu80to120,
    ]

mc_qcd = []
mc_qcd += mc_qcd_em
mc_qcd += mc_qcd_mu

