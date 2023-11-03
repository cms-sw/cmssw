import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

import sys

process = cms.Process('MANUALALCARECO', eras.Run3)

#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    FailPath = cms.untracked.vstring('Type Mismatch') # not crashing on this exception type
    )
options = VarParsing.VarParsing ('analysis')
options.register('outputFileName',
                'PPS_ALCARECO.root',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('alignmentXMLName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignment XML file name")
options.register('alignmentDBName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignmend DB file name")

options.register('globalTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "GT to use")
options.parseArguments()

#SETUP LOGGER
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( 
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(100),
            limit = cms.untracked.int32(50000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring(
        "FwkReport"
    ),
)


#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#SETUP GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

# Handle alignment inputs
if options.alignmentXMLName and options.alignmentDBName:
    print('ERROR: Both alignment XML and DB files specified. Please specify only one.')
    sys.exit(1)

if options.alignmentXMLName:
    # Load alignments from XML  
    print('Loading alignment from XML file:', options.alignmentXMLName)
    process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
    process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(options.alignmentXMLName)
    process.esPreferLocalAlignment = cms.ESPrefer("CTPPSRPAlignmentCorrectionsDataESSourceXML", "ctppsRPAlignmentCorrectionsDataESSourceXML")
elif options.alignmentDBName:
    # Load alignments from DB file
    print('Loading alignment from DB file:', options.alignmentDBName)
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(
            record = cms.string("CTPPSRPAlignmentCorrectionsDataRcd"),
            tag = cms.string("CTPPSRPAlignment_real"),
            connect = cms.string("sqlite_file:"+options.alignmentDBName)
        )
    )
else: 
    print('Using alignment from GT.')

if len(options.inputFiles) != 0:
    inputFiles = cms.untracked.vstring(options.inputFiles)
else:
    # Example input file
    inputFiles = cms.untracked.vstring(
        [
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3581759b-c29a-4422-ac31-7a14c172846f.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ea6494e5-8ec1-40b7-985d-adc1199694a1.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/619d8e3a-cd71-4ad6-a097-71839b09872e.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/7aee80ac-8267-4083-a080-5c52637ad41a.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/f217c448-5af8-4a11-a2ae-b98e375c71bd.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/43e4b18b-8a0f-4a86-9383-4508cc6f89fd.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/593794e5-52a0-416a-a5ab-8d8f7f8b412b.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/119abb80-8c33-4d79-8fbc-29496e4d936c.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/94b4c5f7-a4a0-425f-b28a-378216a9d0e1.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/5c0ffdd8-bfbe-4259-896e-dca1c38b5b88.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/badaf69a-f25a-40b3-8eeb-3f424a7a90b8.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ff83451f-6e87-4775-8bbe-2236a366705c.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/583f4519-266e-438f-82f3-ef97ccdeb9d5.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/69636e72-a932-456f-97c4-bb400fd1151d.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/cf4cb243-d83e-46ec-ba5c-7570fc398039.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/9344194c-4e39-4cf5-9bec-c08fc7bb3d64.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/733386c7-9eff-4ee6-83c2-10eeb88db407.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/9bd4f0a9-bb34-4205-a071-d055ab2ac874.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/328e11d9-aee6-4d0a-8560-130d2cc9f827.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/4131d11c-7570-40b7-8bda-92753c9ffbc7.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/eb2ea9c7-41d1-4164-a385-e2df1025e20e.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/360c09d7-059d-4caf-a1f1-31439e15b78d.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/061457ae-f868-40d1-8618-d4efdc433059.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/6c786fc1-dea1-4867-ae0f-bed38d4ec40e.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/867d956e-388c-42eb-8a99-81aeab936040.root",
# "/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3cbc054d-7f6c-42f5-9bcf-00569c8113bf.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/0d34cb14-b90b-4555-884d-ebcfd6148847.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/241f6401-3bde-4d01-ab01-82eedd598449.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e539dadc-2138-48dc-8f24-d370e0caf045.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c2c9f179-6192-408f-a85f-d06f6914d7a9.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/b1a837c4-7caf-4ab4-ba2f-af518bcdd6b6.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/342c0790-6889-4ad9-ae58-347db33e219c.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/b710965a-f80f-43aa-b56a-260cfe30a9f0.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/7209d01b-3d30-47ed-857f-53dcd6a8b954.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/646a23b9-2d7c-4d00-830c-ff3ff1fd5393.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/54ea426e-b98b-4b8c-85bb-eb8aed796f99.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/18d1da9d-3024-4714-8121-b158094b3484.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e2e13d9a-dc57-4ac0-b6d5-b8e5a6fd6bf1.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ed3f16b3-fdb7-491c-a091-71c990428b74.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/8f5d1100-727b-47c9-a5fc-dee8f7508eb6.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/d2d9797b-7407-4c3b-8f0d-499fa3a57731.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/d9bc82c5-ce0a-4d21-98bf-077bca765189.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/29e66670-aaed-49b8-b93f-fddf8156f63b.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/75d76cc3-7d31-45cf-8fe8-562e06229d46.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/f360bb88-70ad-4c04-bd96-af846169e527.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/a31ecdd8-e566-4653-8e67-f0200d4fd5c0.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/2a3361b8-dffa-4182-aa0e-86470501b53e.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/15e8a907-46f7-465a-af9c-4f4780daa0e8.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/925efb6d-9147-4629-bfd0-8b74a8f8bc42.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/f7aa0b61-980c-4d35-aeaa-d0daa4331780.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/8e3c071e-2ac7-42b4-8eb5-531d6337160e.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/486041e8-9472-4bfd-b0df-9e2f312a7bfb.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/32dd11ab-f8cc-41ba-9546-01d111451314.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/002c312b-b1b5-4f53-8c90-3f52a3f49630.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/d3bc5356-105a-4ce6-ba18-85e3c0e6adaf.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/89a8e5cd-4cc3-403e-a0fa-86bc89ef59f2.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/63500cd7-d8b1-4fca-bee6-39f654f79782.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/38ce74bf-3272-4ba3-80d3-091ce1828914.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3704c43a-92a0-44d7-a840-972bbb4746d8.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/77618b74-b944-4723-88ff-0e4db75c5caf.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/5df850fe-592e-4f9c-8ae8-77a1ac4dc8ea.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e47c2a45-02a6-4de2-b3cf-7f5ae1a2c7c5.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c71d63d7-0d09-4196-99a5-0bad9c7f0743.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/8d2e86f2-76b7-470a-a889-0ec4e01a44c0.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ac85aad7-71ba-4c20-bbdb-712ed12549c9.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/d79f9188-c4c6-4bab-99a2-091846e262c8.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/01d94780-e207-4d95-9537-8577ae410197.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/b47791b3-7530-4b8a-9113-060d9d96a295.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ca44c877-9bc8-4d36-984c-9c4c6c3b312d.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/a6c344fb-af25-4be9-b45f-fa6ba9c57e47.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/dca78ef0-c845-49f5-bf2c-e94167045906.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/afe43975-aeac-43ce-b0b3-4319d3177b32.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/bf60e498-51b5-4197-9039-b393d849aaf3.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/08cf94b2-7bf7-484e-a68b-2e4514b35890.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/0fc61d48-4205-4546-8fe7-cbc2d0c02c8a.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/b9062f02-b3e0-4abd-a911-ef9d6142f617.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/52c4bbab-8370-4875-b29b-03357fec2e81.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/07f103ee-43fb-4e84-860e-4e6d67c76128.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c5564af5-cd6a-457d-92b1-a0269431edb1.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e87cf0df-5e43-4ea7-acd7-ea69a32be341.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/7ecfac9b-711d-4fbb-91e4-e08ed9f1a6e8.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/94fb352e-a5a7-4a6a-b5b9-f9f3771f5ab0.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c5b82663-7ac8-468e-986d-77a849083aae.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/193fa6b9-766f-4a88-b285-b1b99016d207.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e8cece71-a434-4fa0-9299-f4c17006ec5c.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/38f49aa5-9b41-41e2-b9fa-93b3ac51e80e.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/d682a4e5-7990-42cb-a99b-4f39a1aafe5d.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/0c843d1c-cdda-4602-b1c0-72b0ce31a071.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/9e86670e-b4c0-4130-889f-e06a5a0d9695.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/ebadbf88-b809-4464-ae82-f94567115821.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c5abd4b4-7181-48ec-ae03-663ece341f45.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/e292a3b7-c298-4231-b08b-3166f2fc2d23.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/de455de1-4e54-4bc6-a3af-610c17194e30.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/c8dc7585-fda5-45b2-880e-005ff6cd5c6d.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/7b389fe2-a08b-4559-9932-34b21f17b675.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/b6e11cbb-55a1-46a5-86e1-d1d699eff876.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/00b75b3c-dda5-42b3-b09f-d2eeeba1137d.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/120f7d83-dcdf-4211-911a-0d244692d6e7.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/949a8a8c-ec1e-4a00-825d-3da09ea9390f.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/91e7748f-824c-443e-96ee-b7de6be3bd10.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/7dcb9c86-ba69-483b-bcd8-d22ccc0bf8fa.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/57299113-dcb8-44f6-a67f-a94da99c9f36.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/afc552d2-f418-4365-8cdd-8c0a18c425df.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/370ff53d-4b9a-44b6-9f01-9ba6512ffce4.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/f2236e81-8b70-483d-a3a5-4a30cec99001.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/396b7f6f-b6cc-47ca-b9c3-df9ebb58eb83.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/4c691a89-e18d-4001-a6fe-1813bced5e1b.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3c3959c9-6c9b-48d9-a6d6-ec0b701c8e54.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/31e74705-9ff8-4e69-9127-6a685a46109c.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/6fe7ce11-3806-4b3e-b764-73fe022fc743.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/f2d707b8-e75e-4dbd-83e5-b7aa8b962b68.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/3a2b7f06-5153-4c41-ae93-af541f6e21e4.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/379eb8a4-895a-4963-826f-9f8c1573a49e.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/135b7bb0-1e67-4950-aea4-ce9ab1764c0a.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/56164666-12b9-4400-88ef-f5e421836ce7.root",
"/store/data/Run2023C/ZeroBias/AOD/PromptReco-v4/000/368/454/00000/323590e9-36c6-4df0-babf-43908b4978ea.root",
        ]
    )
    
#SETUP INPUT
print('Input files:\n',inputFiles, sep='')
process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    # Drop everything from the prompt alcareco besides the digis at input
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_gtStage2Digis_*_*',
        'keep *_*Digi*_*_RECO'
    )
)

if options.jsonFileName:
    import FWCore.PythonUtilities.LumiList as LumiList
    jsonFileName = options.jsonFileName
    print("Using JSON file:",jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(options.outputFileName),
    # Keep only the new products
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_gtStage2Digis_*_*',
    )
)

# Load the ALCARECO reco step from DIGI
process.load("Calibration.PPSAlCaRecoProducer.ALCARECOPPSCalMaxTracks_cff")
# Remove sampic reco
process.recoPPSSequenceAlCaRecoProducer.remove(process.diamondSampicLocalReconstructionTaskAlCaRecoProducer)

# Adapt ALCARECO InputTags to Offline products
process.ctppsPixelClustersAlCaRecoProducer.tag = cms.InputTag("ctppsPixelDigis","","RECO")
process.ctppsDiamondRecHitsAlCaRecoProducer.digiTag = cms.InputTag("ctppsDiamondRawToDigi","TimingDiamond","RECO")

# processing sequences
process.path = cms.Path(
    process.recoPPSSequenceAlCaRecoProducer
)

process.end_path = cms.EndPath(
  process.output
)