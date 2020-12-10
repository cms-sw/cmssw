import FWCore.ParameterSet.Config as cms

MEMErrorTypes = [
    'TOWERID',
    'BLOCKSIZE',
    'CHID',
    'GAIN'
]

ecalPNDiodeTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        OccupancySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT PN digi occupancy summary'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.')
        ),
        MEMErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/MEM/IntegrityTask MEMErrors'),
            kind = cms.untracked.string('TH2F'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(55),
                nbins = cms.untracked.int32(108),
                low = cms.untracked.double(1),
            ),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.5),
                nbins = cms.untracked.int32(4),
                low = cms.untracked.double(-0.5),
                labels = cms.untracked.vstring(MEMErrorTypes)
            ),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Integrity error and error type counter for MEM boxes. Each x-axis tick corresponds to one SuperModule (SM) as indexed by DCC Id and contains two bins corresponding to the MEM boxes (DCC tower Ids = 69, 70). Nominally, this plot should be empty. Mapping from DCC Id to SM name appears below.<br/><pre>01:EE-07  19:EB-10  37:EB+10<br/>02:EE-08  20:EB-11  38:EB+11<br/>03:EE-09  21:EB-12  39:EB+12<br/>04:EE-01  22:EB-13  40:EB+13<br/>05:EE-02  23:EB-14  41:EB+14<br/>06:EE-03  24:EB-15  42:EB+15<br/>07:EE-04  25:EB-16  43:EB+16<br/>08:EE-05  26:EB-17  44:EB+17<br/>09:EE-06  27:EB-18  45:EB+18<br/>10:EB-01  28:EB+01  46:EE+07<br/>11:EB-02  29:EB+02  47:EE+08<br/>12:EB-03  30:EB+03  48:EE+09<br/>13:EB-04  31:EB+04  49:EE+01<br/>14:EB-05  32:EB+05  50:EE+02<br/>15:EB-06  33:EB+06  51:EE+03<br/>16:EB-07  34:EB+07  52:EE+04<br/>17:EB-08  35:EB+08  53:EE+05<br/>18:EB-09  36:EB+09  54:EE+06</pre>')
        ),
        MEMTowerId = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/MEMTowerId/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemTTId/%(prefix)sIT MemTTId %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        MEMBlockSize = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/MEMBlockSize/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemSize/%(prefix)sIT MemSize %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        MEMChId = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/MEMChId/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemChId/%(prefix)sIT MemChId %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.')
        ),
        MEMGain = cms.untracked.PSet(
#            path = cms.untracked.string('Ecal/Errors/Integrity/MEMGain/'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemGain/%(prefix)sIT MemGain %(sm)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('')
        ),
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/PN/%(prefix)sPOT PN pedestal %(sm)s G16'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Presample mean of PN signals.')
        )
    )
)
