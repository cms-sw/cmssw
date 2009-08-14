import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")
process.load("DQM.CSCMonitorModule.test.csc_daq_info_cfi")
process.load("DQM.CSCMonitorModule.test.csc_dcs_info_cfi")
process.load("DQM.CSCMonitorModule.test.csc_certification_info_cfi")

#-------------------------------------------------
# Offline DQM Module Configuration
#-------------------------------------------------

process.load("DQMOffline.Muon.CSCMonitor_cfi")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.csc2DRecHits.readBadChambers = cms.bool(False)

#----------------------------
# Event Source
#-----------------------------

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("PoolSource",
  fileNames  = cms.untracked.vstring(

    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/FCCD4FD2-0678-DE11-9092-001D09F2543D.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/FC41AA77-2478-DE11-A5ED-000423D99A8E.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F674FE1E-0678-DE11-8C6C-0019B9F704D6.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F49FE2AA-2178-DE11-B178-000423D98FBC.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F4935B22-0678-DE11-852C-001D09F27067.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F48C4CD4-1E78-DE11-BA12-001D09F251FE.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F478655F-1678-DE11-997A-001D09F25438.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F0F5B31A-0C78-DE11-A2D7-001D09F24D8A.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/F0BF966A-0578-DE11-BDBC-001D09F295A1.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/EC3BE2F4-2078-DE11-BFD2-000423D9890C.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/EAB4568F-1378-DE11-A16C-001D09F2910A.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/EAAE779A-1A78-DE11-8849-0030487D0D3A.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/E634AAEB-1978-DE11-BCE0-000423D99AAA.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/E469F980-0C78-DE11-B10D-001D09F2532F.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/E2E89B1F-1278-DE11-AC98-001D09F23174.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/E200BDAD-0978-DE11-AB01-001D09F28EC1.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/E0339730-0D78-DE11-A794-001D09F24D67.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/DEC3FDF7-1478-DE11-9E5A-0019B9F72CE5.root',
    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/DCD5B4F9-1478-DE11-A855-001D09F24EC0.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/DA9A18D5-1278-DE11-8115-001D09F2915A.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/DA8A6AD4-1278-DE11-A449-001D09F290BF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/DA0C0E30-1978-DE11-8BDC-001D09F24047.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/D8918B1F-0678-DE11-B4DE-001D09F2AF1E.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/D6664FED-1978-DE11-BC3B-001D09F295FB.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/D47E8586-0778-DE11-AEE4-001D09F24691.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/D27C5FD7-1278-DE11-BA23-001D09F26509.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CEB5ECF6-0878-DE11-BD1B-001D09F248FD.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CE75B80C-2678-DE11-858B-000423D98EC4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CC906F81-1878-DE11-B4AE-001D09F248FD.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CA92120D-1C78-DE11-9BB0-0030487C6062.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CA470805-1078-DE11-A59A-001D09F25217.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/CA06FC2F-1978-DE11-A915-001D09F244DE.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C6D578D2-0678-DE11-A31A-001D09F241D2.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C65CB9C7-1778-DE11-BAFA-001D09F23A3E.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C4E451F7-1478-DE11-8115-001D09F250AF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C22CD086-0778-DE11-AE40-0019B9F70468.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C0677BC9-1778-DE11-9D04-001D09F297EF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/C02054F1-0878-DE11-93D1-001D09F24FBA.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BE9E626A-0578-DE11-9D00-001D09F253D4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BE14C905-1078-DE11-A45B-0030487BC68E.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BCD702AD-0978-DE11-B422-001D09F2503C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BC46EBF6-0878-DE11-8C7B-001D09F24498.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BC06B6B9-1078-DE11-8F28-000423D94A04.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/BABE5980-1878-DE11-B422-001D09F2512C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B432871D-1778-DE11-AAE4-001D09F25393.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B2FC26D6-1E78-DE11-95CC-000423D9A212.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B2EB02F5-2078-DE11-A819-000423D996B4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B2EA0825-1E78-DE11-8369-001D09F26C5C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B2285439-0878-DE11-93D6-001D09F28F11.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B0E890A9-1578-DE11-9901-001D09F250AF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/B0BBC3E7-0D78-DE11-AFDF-001D09F29321.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/ACFE5B8B-1F78-DE11-A9E2-000423D98E54.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/A83C8607-1C78-DE11-8019-001D09F24934.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/A660195B-2278-DE11-9DEA-000423D98634.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/A2A7B9C8-1778-DE11-8850-001D09F2426D.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/A0C0527E-0C78-DE11-8F62-001D09F251B8.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/9EB06E6C-0578-DE11-8B16-0019B9F6C674.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/9CD46457-1B78-DE11-91DE-001D09F242EF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/9863FC73-1D78-DE11-8155-001D09F25401.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/94C91722-0678-DE11-A8B5-001D09F2AD4D.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/945C2F55-0A78-DE11-B401-0030487C5CFA.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/929D9352-0A78-DE11-8F31-0030487A3232.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/92650C1A-0C78-DE11-B2D7-001D09F2423B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/90D7DF80-0C78-DE11-ADCC-000423D99394.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/90BCD838-0878-DE11-A385-001D09F2A690.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8E787199-0E78-DE11-ADFA-001D09F241F0.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8C90EC4D-0F78-DE11-BAC4-001D09F2546F.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8C1AC881-0C78-DE11-ACB6-001D09F241D2.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8AA028C3-0478-DE11-B75A-0019B9F581C9.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8A8A2401-0478-DE11-95EC-001D09F2960F.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8A77A843-1478-DE11-8402-001D09F292D1.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8A135B70-1178-DE11-8CDA-001D09F23A6B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/8630A8A9-2178-DE11-A6DF-000423D94E1C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/84D27C30-0D78-DE11-8D55-001D09F24489.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/84C32A60-1678-DE11-8A68-0019B9F730D2.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/84659FAB-0978-DE11-83C1-0019B9F581C9.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/808663D2-0678-DE11-8342-001D09F25393.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7EB39B6B-0578-DE11-8044-001D09F24FEC.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7E60918B-1F78-DE11-95D1-000423D95030.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7E405495-0E78-DE11-9CDE-000423D98EC8.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7E09AF5F-1678-DE11-B145-001D09F248F8.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7A94172E-0D78-DE11-BA72-001D09F29533.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7A3E0C0D-1C78-DE11-81F6-001D09F24EAC.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/74A47083-1878-DE11-9136-001D09F29169.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/74493820-2378-DE11-BECF-001D09F24303.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/7264A28C-1F78-DE11-AA32-000423D60FF6.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/706D91D5-1278-DE11-A29B-0019B9F707D8.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/6A302BF0-1978-DE11-855C-0019B9F72CE5.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/68D7B16A-0578-DE11-A67F-001D09F29114.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/685AAB4D-0F78-DE11-8D27-001D09F2447F.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/66137886-0778-DE11-A858-001D09F28D4A.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/64BAA098-0E78-DE11-97C1-000423D6C8E6.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/624F6F1F-1778-DE11-BC55-001D09F25460.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/622FBE8F-1378-DE11-8942-001D09F2516D.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/60D4040E-0B78-DE11-925D-000423D6CAF2.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/5ED54628-1E78-DE11-99B6-000423D9870C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/5E0311B6-1078-DE11-B78F-000423D98A44.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/5A670BF8-1478-DE11-8090-001D09F25442.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/589618CF-1778-DE11-875A-001D09F248FD.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/54C94D9F-1A78-DE11-96EE-00304879FA4A.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/504AD11F-1278-DE11-9985-001D09F2437B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/4E6326F4-0878-DE11-BEB8-001D09F24682.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/4C08D1F6-0878-DE11-B021-001D09F23E53.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/4AA0A2B7-1C78-DE11-9EA7-001D09F2A690.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/4A205639-0878-DE11-B435-001D09F291D2.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/480EB456-0F78-DE11-B679-0019B9F71A6B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/46442774-1D78-DE11-BF95-001D09F25208.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/44369811-0B78-DE11-8AAA-0030487A18A4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/42426920-1278-DE11-A8BA-0019B9F72BAA.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/42189B82-0C78-DE11-82F7-001D09F23A20.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/409D977C-2478-DE11-962B-000423D99160.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/3EB68D8F-1378-DE11-A904-001D09F2960F.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/3E63D9E7-0D78-DE11-B271-001D09F231B0.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/3E16D777-1D78-DE11-A1C8-0019B9F72BFF.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/3C2D69D4-1E78-DE11-8FA7-001D09F2905B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/3AF72A55-0A78-DE11-B7C6-0030487C6062.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/38E70661-1678-DE11-95F4-001D09F24664.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/385481F8-1478-DE11-92DE-001D09F25109.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/36B1A35B-1678-DE11-B018-000423D9870C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/34F37571-1178-DE11-AAB1-001D09F2525D.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/32BB3899-1A78-DE11-8524-0030487C6090.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/320D2EAE-2778-DE11-BB1D-000423D98868.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/2E0130ED-1978-DE11-AA7B-001D09F25041.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/2CE48DA8-1578-DE11-9CE5-001D09F2527B.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/267FC9B7-1C78-DE11-B0D3-001D09F2503C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/249138E3-0D78-DE11-A65A-000423D98DD4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/243E960D-0B78-DE11-947B-001D09F2532F.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/1EBD7601-0478-DE11-AD46-001D09F251D1.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/1CD46EF5-2078-DE11-9117-000423D99658.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/1AA51DB5-1078-DE11-A924-000423D8F63C.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/14E2883F-2078-DE11-A2AA-000423D99CEE.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/12E2CC1F-1278-DE11-8A07-001D09F28EA3.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/121E6007-1078-DE11-81A0-001D09F290CE.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0AF3CB24-1E78-DE11-B10A-000423D6BA18.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0AB0E90A-1C78-DE11-B7C7-001D09F28D54.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0A641B53-1B78-DE11-94BC-001D09F241B4.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0A568C05-1078-DE11-A846-001D09F2424A.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/08D8DA4E-2578-DE11-AE90-000423D99E46.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/085C005C-2878-DE11-A66D-001D09F251BD.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0691BBA8-1578-DE11-BDEC-001D09F291D7.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/063E38C8-1778-DE11-BE6B-001D09F232B9.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/04DB8CAB-2178-DE11-B8AA-000423D944F0.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/0270FBB2-0478-DE11-8D95-001D09F292D1.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/02433F0C-1C78-DE11-85DC-001D09F23A02.root',
#    '/store/data/Commissioning09/Cosmics/RAW/v3/000/108/298/00DFD0F4-2078-DE11-8FA6-000423D98BE8.root'

  ),
  #skipEvents = cms.untracked.uint32(1129)
)

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
process.DQMStore.referenceFileName = '/afs/cern.ch/user/v/valdo/data/csc_reference.root'
#process.DQMStore.referenceFileName = '/nfshome0/valdo/CMSSW_2_1_0/src/DQM/CSCMonitorModule/data/csc_reference.root'

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
process.dqmSaver.dirName = '.'

#-----------------------------
# Magnetic Field
#-----------------------------

process.load("Configuration/StandardSequences/MagneticField_cff")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#process.GlobalTag.connect = "sqlite_file:/nfshome0/malgeri/public/globtag/CRZT210_V1H.db"
#process.GlobalTag.connect = "frontier://FrontierDev/CMS_COND_CSC"
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRZT210_V1H::All"
#process.GlobalTag.globaltag = 'CRAFT_V3P::All'
#process.GlobalTag.globaltag = "CRAFT_30X::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_V17H::All"
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = 'GR09_31X_V1H::All' 
process.GlobalTag.globaltag = 'GR09_31X_V1P::All' 
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


#--------------------------
# Web Service
#--------------------------

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Message Logger
#--------------------------

MessageLogger = cms.Service("MessageLogger",

# suppressInfo = cms.untracked.vstring('source'),
# suppressInfo = cms.untracked.vstring('*'),

  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
#    WARNING = cms.untracked.PSet(
#      limit = cms.untracked.int32(0)
#    ),
#    noLineBreaks = cms.untracked.bool(False)
  ),

  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
  ),

#  critical = cms.untracked.PSet(
#    threshold = cms.untracked.string('ERROR')
#  ),

  debugModules = cms.untracked.vstring('*'),

  destinations = cms.untracked.vstring(
    'detailedInfo', 
    'critical', 
    'cout'
  )

)

#--------------------------
# Sequences
#--------------------------

process.p = cms.Path(process.dqmCSCClient * process.cscDaqInfo * process.cscDcsInfo * process.cscCertificationInfo + process.dqmEnv + process.dqmSaver)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)


