import FWCore.ParameterSet.Config as cms

process = cms.Process("TIMECALIBANALYSISELE")

filelist = cms.untracked.vstring()
filelist.extend([
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/003EC246-5E67-E211-B103-00259059642E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0045FFC7-6167-E211-9247-003048678B34.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/004777AA-9B67-E211-80B4-003048FFCBA4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0081344B-7C67-E211-9A3D-002618943831.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/00AC7ED8-4C67-E211-A5F9-0025905964C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/00D947F0-A267-E211-8224-002618943985.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0279EA69-6267-E211-888B-002590593872.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02900832-9267-E211-B430-002618943821.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02A18685-5567-E211-9D79-00248C0BE01E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02A42D5B-7B67-E211-8463-0026189438A2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02B11DEA-9967-E211-9C71-002618FDA262.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02BD2ACE-6567-E211-9326-003048679010.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02CE0229-A467-E211-ABD1-00248C55CC40.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/02DB720E-4767-E211-A94A-00304867C0EA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/041FCB57-6E67-E211-B876-0025905938A4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/04267109-6567-E211-95C9-00248C0BE005.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/062A0C12-6167-E211-8114-002354EF3BE1.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/062BC8DC-6067-E211-BF2F-003048FFD752.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0655FC24-8367-E211-9D95-003048679266.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/068C6CE5-5267-E211-9085-002618943869.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/068E24EC-5F67-E211-AE24-002618943962.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/06A06342-5E67-E211-8F2E-003048FFD744.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/06C02608-5667-E211-809D-002618943894.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/06C720DF-5F67-E211-AF21-002590596498.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0845F321-8467-E211-A947-003048678F74.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0870D45D-4667-E211-9E45-002618943838.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/08AEDBB8-5F67-E211-91D0-002618943950.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0A05F066-6267-E211-B9A7-003048D42D92.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0A290993-A367-E211-A739-002618943978.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0AD79DBD-8667-E211-9B86-00304867920C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0AEDD846-5767-E211-9BCE-0026189438CF.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0C33C0B0-6367-E211-9752-0025905938B4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0CCAAF0E-6167-E211-8ED1-002618FDA248.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0E1FDE8F-5F67-E211-94CE-002618943807.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0E533104-8267-E211-AB17-003048678F8C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0E713620-8467-E211-B216-0026189437F8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0EA9B35A-8F67-E211-B4FE-003048679236.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0EB69CD7-8567-E211-A1A6-003048FF9AA6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0EB7873A-5567-E211-A3AE-003048679012.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/0EDF2A1F-6967-E211-95B6-00261894384F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10510048-6A67-E211-A909-00261894397B.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10624D49-A667-E211-B58B-0025905938A4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10785FA5-F967-E211-B234-002354EF3BE3.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10AB4AC9-6267-E211-BC7F-002590596490.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10C8A92C-8867-E211-B5C7-003048678B0E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/10FFB119-6167-E211-BEDF-003048678C62.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/121E02B6-4E67-E211-B11D-0025905938A4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/123305FC-9067-E211-B0D4-00304867BEDE.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/12368FD4-8567-E211-87C6-00261894393D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/128D9D78-9567-E211-89EE-0026189438A0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/12905568-4A67-E211-A437-003048FFD71E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/129A98C8-6567-E211-859B-00261894398A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/14417594-8C67-E211-9DDF-002618943984.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/144588C5-6767-E211-8DD3-003048FFD730.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/14735A59-7467-E211-ACC0-00304867901A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/14BE414A-A667-E211-886D-002618943933.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/14D4A753-8F67-E211-8B19-003048678E52.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/14E6D204-7967-E211-8233-002618943949.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/16062EFD-7B67-E211-8691-0025905938A4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/16C46AD4-C267-E211-BF66-003048FFCC2C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/16E7BC11-1068-E211-B137-003048FFD71E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/180608D3-6067-E211-BB8E-0026189438BA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/180F1DE8-7867-E211-8CA5-002590596468.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/181838A1-9167-E211-A85A-003048679296.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/182B6D21-8367-E211-9B5C-003048FF86CA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/183A19FB-7067-E211-8021-003048678B18.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/185B0715-7067-E211-B8DF-00261894385A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/185CD345-5E67-E211-B558-0025905964A6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/185D8BFF-9067-E211-AC49-0025905964B4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/187F030B-7867-E211-9E7A-0026189437FC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1893888A-5567-E211-A7A0-0025905938A8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/18EC18EE-6C67-E211-A388-00259059391E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/18FAB224-9867-E211-8592-002590593876.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1A05C8C5-6B67-E211-90C8-002618943978.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1A1B1857-8D67-E211-97FC-0030486790C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1A51C1A3-6167-E211-A06F-0030486792B6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1A5529FB-7F67-E211-B53B-0026189437FA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1AA63687-A467-E211-A724-002618943982.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1C2D3AF2-7567-E211-A1B0-002618943925.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1C6A2F91-9567-E211-AB3A-00304867924A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1C7B61DB-8A67-E211-B94D-00304867BFA8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1C80F951-8B67-E211-B6B7-002590596490.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1CB88E5E-8E67-E211-947D-00261894389F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1CDBEF5D-7467-E211-94CC-003048FFCBFC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1CDDEE49-4B67-E211-A8A5-0025905964B4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1E056BFC-7E67-E211-A31D-0026189438CF.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1E4336C0-6B67-E211-8E88-0025905938AA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1E5CC98C-5267-E211-8F26-002618943925.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/1EB46FF6-6267-E211-B752-002618943985.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/20085342-8867-E211-8194-003048FFD760.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/204C61AE-7967-E211-9985-003048678C3A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/204E5895-7667-E211-B1A0-003048678B30.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2053E1A1-9567-E211-B600-00248C0BE014.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2072FCB5-9767-E211-931D-002618943951.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/209D49D2-8567-E211-9C1C-002618943800.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/20C219AD-AB67-E211-9C2E-003048678E8A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/20CBAC07-5667-E211-B179-002618943983.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/222A352E-5567-E211-8358-002618943953.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/22375641-8C67-E211-B265-003048679214.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/241B4AD2-6267-E211-B166-003048678EE2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/244DDAE7-8567-E211-9482-0026189438E4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2471BF54-7467-E211-B733-00261894389A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/24A2CEA8-6C67-E211-8431-003048FFD770.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/24B0B94D-6A67-E211-BB69-003048FFCB9E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/24B384D8-4867-E211-8170-002618FDA277.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/24C16B40-8C67-E211-AE21-002618943948.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/24F1441D-6F67-E211-956E-0026189438C2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/26517260-C467-E211-89D9-003048FF86CA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/26B7B448-5767-E211-B39D-002618943980.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/281D7AC6-6067-E211-8364-003048678F9C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/28589BAF-5668-E211-8780-00261894386D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/28A41059-7467-E211-A06F-0025905938B4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/28CA2FF3-7167-E211-B0B1-003048FFCB74.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A191A24-8367-E211-8169-0025905938A8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A2F3CF8-9967-E211-A810-003048FFD796.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A35ACF9-7267-E211-A1B9-0026189437FA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A62B090-5F67-E211-B8A0-0026189437F8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A7FA077-6667-E211-9A1B-003048FF9AC6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2A8AF23E-6067-E211-B537-0030486792B6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2ADE5EC0-B467-E211-87FC-002618FDA263.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2C08387B-7267-E211-999F-002354EF3BDC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2C3EBD3F-7567-E211-A628-00304867BFA8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2C912795-7667-E211-8DC8-002618943985.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2CA9BC47-8067-E211-B5F8-0026189438E0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2CC49F04-7F67-E211-B468-003048678AE2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2E0064D8-8567-E211-BB23-003048678C3A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2E25E281-8967-E211-9481-003048678C06.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2E4DE0BF-8467-E211-84F0-002618943886.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/2EED53E2-8567-E211-BE40-002618943937.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/300D691D-6967-E211-A4FF-003048678B0A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/300FDD59-7F67-E211-8E9C-003048678BAA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3028917F-6467-E211-8FFA-002590596484.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/30302C02-7B67-E211-B691-00248C55CC9D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/304D24E6-8B67-E211-8FC3-002618FDA248.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3065460F-6567-E211-BCCA-003048FFCB9E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/309CF2FB-8067-E211-874E-00304867BF18.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/30CE71C6-9167-E211-B2E5-002618943854.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/30FB3E54-8C67-E211-9A7E-003048679244.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/30FC5EC0-6367-E211-BA30-00259059642E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3229DDCA-6567-E211-9B94-003048FFD7C2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/32743F0C-6867-E211-9B30-0030486792B6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/329CDD4A-7C67-E211-8BCF-003048FFD754.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/340EE586-8867-E211-9215-00261894388F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/34122263-5C67-E211-B6F1-0030486790C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/34258541-A867-E211-A346-00304867BEC0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/346FEBBA-9767-E211-ABD5-0025905938AA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/34C868DA-9267-E211-889E-0026189438DA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/34CB49B6-7367-E211-ACC9-002618943829.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/34F3F4AA-7967-E211-BC86-002618943963.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3620DDB5-6667-E211-81D6-002590596498.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/362663D3-BC67-E211-9DC7-002618943882.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/362E420D-5667-E211-BFDF-00261894392D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/366699E3-A467-E211-AB56-0026189438D7.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/367A9746-8067-E211-BC16-00261894398C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/367E9129-8B67-E211-82A5-00261894395A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/368C8BBC-9869-E211-ABEE-002618943867.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/36B5BD05-5667-E211-B638-003048678FE0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/36BDD198-4767-E211-B7F1-003048FFCB6A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/36E34ACF-6267-E211-B923-003048FFD754.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/38D2B48E-8967-E211-9115-003048678C9A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/38EE0AF9-5767-E211-BAC7-003048678B30.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A0725F2-8867-E211-92EE-003048FF86CA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A404792-8867-E211-9B02-003048678B72.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A494B8A-6567-E211-80D3-003048678D86.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A6266FF-4D67-E211-A5B1-003048678AC0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A6F8889-5867-E211-ACC8-00304867924E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A90FCE8-6367-E211-BF07-003048678B44.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3A96DC1C-7767-E211-AC0F-003048678A80.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3AAEB658-8F67-E211-B604-0026189438A9.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3ACDF0F8-7267-E211-AA5A-00261894396A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3ADFBC3B-7567-E211-B542-00261894397D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3AE5C0F9-6B67-E211-9320-00261894391F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3AEE99D9-6D67-E211-B032-00261894391D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3AFFFFFC-7867-E211-A68B-003048678B00.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3C0AC874-8B67-E211-A139-0026189438BF.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3C69A677-9667-E211-B0EF-002618943899.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3C850DB9-6367-E211-82B8-0026189438ED.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3CFBA6BD-6367-E211-BD2D-00259059642E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3E216AF0-7C67-E211-B971-003048679080.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3E5EA14A-8667-E211-BE29-00261894382A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/3EE416DC-8567-E211-8EC6-0030486790B8.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/40350154-8F67-E211-9B8F-002618943932.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4070E11B-7067-E211-90C8-003048FFD732.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/407607FC-9067-E211-AC9A-00261894387E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/40A2C59B-4367-E211-B3EB-00261894393E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/40C5DEAF-4D67-E211-98BC-0026189438CF.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/40C6FA35-9667-E211-A64C-0026189438C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/40FD1DF7-9D67-E211-8B2B-0025905964C4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4210FD7E-7267-E211-817E-003048FFD732.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/421F1C27-9267-E211-8C0B-0030486790B0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/425F4780-6467-E211-8C32-0025905964C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4279C735-4C67-E211-A2BD-0025905964C0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42A85C77-B967-E211-9F6E-0026189438BA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42AE1787-4867-E211-BEEC-003048FFCC18.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42BA066F-8367-E211-9BEA-002590596486.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42D8A8AB-A467-E211-A797-0026189437FE.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42ECFD62-5C67-E211-B3F5-002618943915.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/42FB408C-5E67-E211-B741-002618FDA211.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4407DC86-5667-E211-A9A4-0026189438F6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/440B7973-5067-E211-9012-002618943918.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/445F74B1-5467-E211-8D21-0026189438EB.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/44B89F8F-7E67-E211-BD54-0026189438AE.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/44E8AEE2-7867-E211-A930-003048FFCC18.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4664D092-9267-E211-9060-003048678A6A.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4687655F-8667-E211-B93B-003048D15DDA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/46B416B4-6767-E211-861C-0025905822B6.root',
#'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/46BEA2B4-6767-E211-AB3A-0025905822B6.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/46C8594F-9D67-E211-9053-00261894389E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/46D292E1-8B67-E211-A22F-0025905964BC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/46EB42A0-9567-E211-A35A-003048FFCC1E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/480E3195-7E67-E211-B073-002618943809.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/482075A7-8967-E211-894E-002618943963.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4877518D-8967-E211-82B5-003048FFD760.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/48A743FC-7F67-E211-830A-002618943984.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/48AEAC31-8A67-E211-ACE3-002354EF3BDD.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4A644356-AD67-E211-BEF2-003048679296.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4A9A7E40-7567-E211-B62A-002590596490.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4AD54BC3-1B68-E211-8B30-00248C0BE014.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4AFE44D6-6E67-E211-B368-003048678E52.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4C0174A4-9E67-E211-B644-002618943901.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4C33A3DA-4867-E211-A219-0025905964BC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4C397CFF-B767-E211-AC35-002618943960.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4C50C971-AF67-E211-849B-002618943937.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4C8AC141-7667-E211-8855-002618FDA287.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4CACFC1D-5B67-E211-817B-002618943864.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4CBF1650-6A67-E211-9690-0026189438C9.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4E0D50B5-9E67-E211-B41B-003048FFD7C2.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4E323664-A367-E211-BA65-003048679076.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4E777035-9967-E211-8D00-002618943944.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4EB72E1F-7A67-E211-BBCC-0025905938A4.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4EBFB43E-5E67-E211-B077-002618943953.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4EF10AD9-7767-E211-8728-0026189438DA.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/4EF61C45-D167-E211-9D84-0026189438FC.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/503EE428-A067-E211-AF63-00261894382D.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/5048EF69-6867-E211-9C1C-00248C55CC62.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/506FF5E4-5667-E211-8FC5-003048FFD732.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/50B31E5A-8D67-E211-85FB-002618943933.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/50CDB4C1-8467-E211-A137-002618943867.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/50CE7B73-8367-E211-A735-003048678F8C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/50DE9C56-7467-E211-8480-00261894391F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/50F3E396-8C67-E211-B154-00261894383F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/5250905E-8767-E211-9CDA-00261894386F.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/5250B0B2-8667-E211-8F77-002618943842.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/5276D262-5C67-E211-8564-0026189438AE.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/527B2F44-B467-E211-B846-003048FFCC2C.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/5294713D-8867-E211-8D1D-0030486790B0.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/52CB3F82-8967-E211-AFF1-002618943882.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/52EC9633-8867-E211-943D-00261894386E.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/547AE844-9A67-E211-978A-00248C55CC97.root',
'file:/hdfs/cms/phedex/store/data/Run2012A/DoubleElectron/AOD/22Jan2013-v1/20000/548BD0EF-5F67-E211-AB31-00261894393F.root'



])





# Output - dummy
process.out = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(),
    fileName = cms.untracked.string('file:pippo.root'),
    )


# gfworks: to get clustering 

# Geometry
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi") # gfwork: need this?
process.CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder")


# pat needed to work out electron id/iso
from PhysicsTools.PatAlgos.tools.metTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.pfTools import *

from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import *


# Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'GR_R_53_V18::All' )
# tag below tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'GR_R_42_V14::All'

# this is for jan16 reprocessing - tested in CMSSW_4_3_0_pre3
#process.GlobalTag.globaltag = 'FT_R_42_V24::All'

process.load('Configuration.StandardSequences.MagneticField_38T_cff')


# Trigger
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()



#------------------
#Load PAT sequences
process.load("PhysicsTools.PatAlgos.patSequences_cff")
process.load("PhysicsTools.PatAlgos.tools.pfTools")
#
## THis is NOT MC => remove matching
removeMCMatching(process, ['All'])
#
#
## bugfix for DATA Run2011 (begin)
removeSpecificPATObjects( process, ['Taus'] )
process.patDefaultSequence.remove( process.patTaus )

#
###
process.patElectrons.isoDeposits = cms.PSet()
#
process.patElectrons.addElectronID = cms.bool(True)
process.patElectrons.electronIDSources = cms.PSet(
        simpleEleId95relIso= cms.InputTag("simpleEleId95relIso"),
            simpleEleId90relIso= cms.InputTag("simpleEleId90relIso"),
            simpleEleId85relIso= cms.InputTag("simpleEleId85relIso"),
            simpleEleId80relIso= cms.InputTag("simpleEleId80relIso"),
            simpleEleId70relIso= cms.InputTag("simpleEleId70relIso"),
            simpleEleId60relIso= cms.InputTag("simpleEleId60relIso"),
            simpleEleId95cIso= cms.InputTag("simpleEleId95cIso"),
            simpleEleId90cIso= cms.InputTag("simpleEleId90cIso"),
            simpleEleId85cIso= cms.InputTag("simpleEleId85cIso"),
            simpleEleId80cIso= cms.InputTag("simpleEleId80cIso"),
            simpleEleId70cIso= cms.InputTag("simpleEleId70cIso"),
            simpleEleId60cIso= cms.InputTag("simpleEleId60cIso"),
            )
###
process.load("ElectroWeakAnalysis.WENu.simpleEleIdSequence_cff")
process.patElectronIDs = cms.Sequence(process.simpleEleIdSequence)
process.makePatElectrons = cms.Sequence(process.patElectronIDs *
                                        process.patElectrons)
process.makePatCandidates = cms.Sequence( process.makePatElectrons   )
process.patMyDefaultSequence = cms.Sequence(process.makePatCandidates)



# this is the ntuple producer
process.load("CalibCalorimetry.EcalTiming.ecalTimeEleTree_cfi")
process.ecalTimeEleTree.OutfileName = 'EcalTimeTree'
process.ecalTimeEleTree.muonCollection = cms.InputTag("muons")
process.ecalTimeEleTree.runNum = 999999
#process.ecalTimeTree.endcapSuperClusterCollection = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower","")



process.dumpEvContent = cms.EDAnalyzer("EventContentAnalyzer")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.p = cms.Path(
    process.patMyDefaultSequence *
    # process.dumpEvContent  *
    process.ecalTimeEleTree
    )

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 250

# dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=124020" | grep store | awk '{printf "\"%s\",\n", $1}'
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = filelist,
    #fileNames = cms.untracked.vstring('file:input.root')
    #'/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/135/494/A4C5C9FA-C462-DF11-BC35-003048D45F7A.root',
    #'/store/relval/CMSSW_4_2_0_pre8/EG/RECO/GR_R_42_V7_RelVal_wzEG2010A-v1/0043/069662C9-9A56-E011-9741-0018F3D096D2.root'
    #'/store/data/Run2010A/EG/RECO/v4/000/144/114/EEC21BFA-25B4-DF11-840A-001617DBD5AC.root'

   # 'file:/data/franzoni/data/Run2011A_DoubleElectron_AOD_PromptReco-v4_000_166_946_CE9FBCFF-4B98-E011-A6C3-003048F11C58.root'
 #       'file:/hdfs/cms/phedex/store/data/Run2012C/SinglePhoton/RECO/EXODisplacedPhoton-PromptSkim-v3/000/198/941/00000/0EA7C91A-B8CF-E111-9766-002481E150EA.root'

 #   )
    
 )



