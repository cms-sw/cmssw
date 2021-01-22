import FWCore.ParameterSet.Config as cms

process = cms.Process("rpcTier0SourceProcess")

################# Input ########################

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/FCA6CCE5-F2C2-DE11-A1BB-000423D9A212.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/FC59F5F3-C7C2-DE11-AB94-0030487A1990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/FC394E44-DAC2-DE11-919E-001617C3B6C6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/FC083776-E1C2-DE11-91E8-001617E30D12.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F89E64F2-BEC2-DE11-B984-003048D375AA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F623F5C4-BAC2-DE11-BF8E-001D09F24FEC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F2415DA2-E7C2-DE11-BA54-000423D98B6C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F2313D20-FEC2-DE11-AFB3-000423D94990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F04E02BA-CFC2-DE11-B168-000423D98834.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/F0475EDA-EBC2-DE11-BCE5-000423D998BA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/EEF6D8F3-E6C2-DE11-AA4C-000423D99E46.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/EEE2407C-F1C2-DE11-A4FC-000423D991F0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/ECE73220-FEC2-DE11-A7F9-000423D6006E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/EC69873D-D3C2-DE11-8CB0-0030487A1990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/EC626D04-E2C2-DE11-9E55-001D09F2512C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/EA4E255B-D5C2-DE11-B23B-000423D94524.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E89F19A5-D4C2-DE11-97EA-0030487A1FEC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E885472E-D8C2-DE11-AFC4-000423D9989E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E61B1A33-D6C2-DE11-A3C9-003048D37580.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E4DBF8B2-CDC2-DE11-8F0D-001D09F2906A.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E46E74FB-DAC2-DE11-8756-003048D3750A.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E2D59FEB-B9C2-DE11-BA6D-001D09F2B2CF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E2A34759-C2C2-DE11-9986-001D09F25041.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E2959B5D-B4C2-DE11-B118-001D09F24353.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E251AE30-CCC2-DE11-A0A1-000423D98EC8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E20DF622-D1C2-DE11-AA54-001D09F24DDF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E09CB035-F9C2-DE11-B364-000423D94A20.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E094A157-D5C2-DE11-BD15-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/E07875FD-F4C2-DE11-8470-000423D9880C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/DEC1F652-E1C2-DE11-ADA2-001617DBCF6A.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/DEB4C428-EBC2-DE11-9EC5-000423D98800.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/DC1F764A-B4C2-DE11-92D3-001D09F290CE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/DA0D8A39-C0C2-DE11-A936-000423D98DC4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8E547A8-E0C2-DE11-908F-001D09F25208.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8D555DF-D8C2-DE11-A0A9-000423D94E70.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8BB30CA-F0C2-DE11-AC96-000423D98FBC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8B5B3D1-E4C2-DE11-841B-000423D986A8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8A5D8F8-DFC2-DE11-ACFD-001D09F2905B.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D8771A8F-B3C2-DE11-A49A-001D09F2546F.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D81130B1-F5C2-DE11-906F-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D6F2D788-BFC2-DE11-B8BD-001D09F24E39.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D68DC430-CCC2-DE11-8E8F-000423D99EEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D6856365-F6C2-DE11-828D-000423D98950.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D668014B-F4C2-DE11-B38E-000423D99F1E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D656795E-D5C2-DE11-9A0E-003048D375AA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D0ED3528-EBC2-DE11-AD7C-000423D99EEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/D0379158-E8C2-DE11-9B69-000423D98930.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CEF0E5EA-F4C2-DE11-9C58-000423D991D4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CEE83EC4-CAC2-DE11-9323-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CE914214-DDC2-DE11-936A-000423D6C8EE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CE3C47F2-C7C2-DE11-B92E-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CE295014-DDC2-DE11-985A-000423D98B08.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CCABF0BB-E2C2-DE11-BBAB-001617E30F48.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CC7B593A-D6C2-DE11-8DD4-001D09F29849.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/CC78FCBD-E9C2-DE11-A81F-000423D987E0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C86BDA0D-C3C2-DE11-BB6C-0019B9F705A3.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C828068F-B3C2-DE11-B273-001D09F27003.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C823F2B0-F5C2-DE11-84BE-000423D98634.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C801FEEF-CCC2-DE11-A6C9-003048D373AE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C6533358-D5C2-DE11-8006-001617C3B6C6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C6276566-D0C2-DE11-B9ED-0019DB29C614.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C411BB02-E2C2-DE11-9F86-001D09F23C73.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C0E02F24-D6C2-DE11-9DD7-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C07BB0F8-CCC2-DE11-B0C4-003048D37456.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C0798AA6-BAC2-DE11-BC3A-001D09F24F65.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/C03541EF-F4C2-DE11-8A15-000423D99896.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BEA93C43-F2C2-DE11-B3DA-000423D9890C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BCFF2DE0-D8C2-DE11-903D-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BC703366-E1C2-DE11-B562-001D09F24DDF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BA708659-E8C2-DE11-9579-000423D99AAA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BA47BE0B-F0C2-DE11-B746-000423D8FA38.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BA3CFE66-F6C2-DE11-A3FA-000423D98930.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/BA0C97CE-E4C2-DE11-B0CE-000423D98804.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B8D080AB-D4C2-DE11-9E3C-001617E30D52.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B885A819-B7C2-DE11-9E65-001D09F29146.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B850E01F-D1C2-DE11-AAAF-001D09F25401.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B84D0C7A-F8C2-DE11-BF05-000423D98950.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B8331907-E2C2-DE11-B047-001D09F25456.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B6433E84-D2C2-DE11-B381-000423D6C8E6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B4E3C0AA-DBC2-DE11-8D4F-000423D987FC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B42AEDEC-BCC2-DE11-88A5-001D09F290CE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B2A3A73B-D3C2-DE11-BA82-000423D6B444.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B0807DB3-CDC2-DE11-8A65-001D09F27003.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/B07BEAC8-C8C2-DE11-AA5C-003048D375AA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AE7009F2-D3C2-DE11-913A-001D09F2512C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AE5A86CC-DDC2-DE11-B3C2-0030487A18F2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/ACEC74BF-E9C2-DE11-B53C-000423D9880C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AC94BCCB-F7C2-DE11-9DF5-000423D94494.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AC623E8C-BFC2-DE11-8321-001D09F2426D.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AC4C4A2F-CCC2-DE11-96FC-000423D6B444.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AC4AABE8-F9C2-DE11-95D9-000423D944FC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/AC448DB9-CFC2-DE11-BB48-001D09F29849.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A8D45EB3-CDC2-DE11-BE14-001D09F231C9.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A8941FE5-F2C2-DE11-90D0-000423D94494.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A6835D30-B9C2-DE11-8201-000423D94990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A4E98A4D-23C3-DE11-96C2-0030487D1BCC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A26A7443-F2C2-DE11-8D26-000423D60FF6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A0F43DC2-D6C2-DE11-9E44-000423D6CA6E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/A00BFAA1-E7C2-DE11-9AA3-000423D94C68.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9CC31EC3-DEC2-DE11-966F-0030487A18F2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9C950E1B-F7C2-DE11-9D7B-000423D9A2AE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9C6889F8-DFC2-DE11-BCF2-001D09F2A49C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9C296C1A-F7C2-DE11-BB09-000423D94A20.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9C203BF0-D3C2-DE11-B1D3-001D09F276CF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9AC3DDA2-E7C2-DE11-9B62-000423D99996.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9A793A07-E9C2-DE11-BD96-000423D98800.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9A256437-C5C2-DE11-BAC7-003048D2C020.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9A14D23F-D3C2-DE11-8CAE-001617E30D52.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/98F9D126-F5C2-DE11-A746-000423D98EC4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9875C8CF-F7C2-DE11-9BA6-000423D98C20.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/96D3CF47-F4C2-DE11-8B36-000423D98930.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/9696DB3E-C0C2-DE11-8E9D-000423D990CC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/94ED99CA-B7C2-DE11-877D-001D09F24399.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/94A836C6-F0C2-DE11-9A33-000423D94A20.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/946B9273-0BC3-DE11-9C46-000423D985B0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/945D024C-C7C2-DE11-A1CF-003048D3756A.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/943641B2-EEC2-DE11-B8ED-000423D9517C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/94164665-EDC2-DE11-9B64-000423D98DC4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/92DEDE3A-E6C2-DE11-8214-000423D6CA02.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/92776C04-E2C2-DE11-83B1-001D09F24EE3.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/90B2A168-C9C2-DE11-B141-000423D6CA72.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/904726A9-BAC2-DE11-86B4-001D09F24353.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/902A50B9-CFC2-DE11-A11C-001D09F297EF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/901FF406-BCC2-DE11-A52E-003048D37514.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8EF69DC4-CAC2-DE11-AC1B-000423D94908.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8EE1FEE9-F4C2-DE11-8876-000423D98BC4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8E74E872-EFC2-DE11-8B3F-000423D9A2AE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8CAF2223-D1C2-DE11-B530-0030487A1990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8C90136B-D0C2-DE11-B395-000423D6CA72.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8C856F06-E9C2-DE11-8A31-000423D8FA38.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8A9FB004-E7C2-DE11-94B1-000423D9A212.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/8A251C62-B6C2-DE11-B06A-000423D98B08.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/883364F8-DFC2-DE11-BDF3-001D09F29538.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/86D305E9-B9C2-DE11-93A6-001D09F24F1F.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/86916825-E4C2-DE11-B4A4-0030487C6062.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/84E46631-CAC2-DE11-9342-003048D375AA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/849C73B0-DBC2-DE11-8CE7-000423D99AAE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/845F98D1-BEC2-DE11-9295-003048D37560.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/82FEB962-C9C2-DE11-88C2-000423D6B42C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/82DADEFA-DAC2-DE11-9157-000423D944F0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/82928183-EDC2-DE11-8BF6-000423D944DC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/822FF46D-FDC2-DE11-86BC-000423D99614.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/80EEFE81-E5C2-DE11-AF04-000423D99AA2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/80BF1924-E4C2-DE11-AD9C-000423D94908.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7E6393F2-CCC2-DE11-ABE9-001617C3B5F4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7CEF98A0-F3C2-DE11-8E35-000423D99394.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7C03A392-D9C2-DE11-8CF7-000423D6C8EE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7AA499EF-B9C2-DE11-A34A-001D09F2932B.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7A4EDF77-C4C2-DE11-93A5-001D09F242EA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7A039E67-E3C2-DE11-9309-000423D98B28.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/76C6D86F-D7C2-DE11-9463-001617C3B6C6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/7696647C-F1C2-DE11-A6AB-000423D99AAA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/742485C2-D6C2-DE11-8503-000423D9870C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/72C52153-CEC2-DE11-8676-00304867342C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/728BF467-D0C2-DE11-87E8-001D09F24600.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/72626AB7-CFC2-DE11-8D9D-001D09F276CF.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/721B9B59-D5C2-DE11-B0BA-003048D3750A.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/70E648CB-DDC2-DE11-B321-000423D996C8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/70512CA9-D4C2-DE11-8E85-003048D2BED6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/70316F33-B9C2-DE11-A74E-001617C3B654.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6EEEC267-D0C2-DE11-A368-0019B9F7312C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6EA5735D-DCC2-DE11-9A87-000423D94524.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6E830331-B9C2-DE11-B897-000423D98800.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6C9060D5-C5C2-DE11-A3E7-003048D37560.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6C0E783C-D3C2-DE11-8DCF-0030487A1FEC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6AD95306-0AC3-DE11-82AA-000423D6CA02.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/686796CC-F7C2-DE11-8E81-000423D987E0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/66E7AD45-DAC2-DE11-A1B0-000423D987FC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/66518F61-DCC2-DE11-9F31-000423D98750.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/662E8588-BFC2-DE11-A302-001D09F25401.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/64B314F5-CCC2-DE11-8E7B-003048D374F2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/64B30D8D-D2C2-DE11-987C-000423D952C0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/648A4272-E1C2-DE11-8BA7-001D09F2AF96.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/647F1E10-F0C2-DE11-A933-000423D99394.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/64313667-E3C2-DE11-8F3F-000423D9997E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/62AE5BCA-B7C2-DE11-83A1-001D09F2441B.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/6297F16E-FDC2-DE11-976E-001D09F292D1.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/60F0DC22-E4C2-DE11-BEF8-000423D99AA2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/5EE555F6-CCC2-DE11-9BFB-003048D37560.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/5E8FE45B-EDC2-DE11-84E3-000423D94C68.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/5CCD7DB9-E2C2-DE11-8E98-000423D99AAA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/587DBABC-E2C2-DE11-99AC-001617E30D12.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/58360FE5-F2C2-DE11-A6AB-000423D99996.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/56FD2CB0-DBC2-DE11-B0A4-000423D99E46.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/56C45C23-D1C2-DE11-B3F3-001D09F231C9.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/56A578F3-CCC2-DE11-84B9-0030487A1990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/56589B72-D7C2-DE11-8677-000423D94E70.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/521F52CA-DDC2-DE11-B06D-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/50F3EF75-EFC2-DE11-AD80-000423D98634.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/50B20D65-C9C2-DE11-8C87-001D09F25401.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/50A618F3-B4C2-DE11-B2A5-000423D98EC4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/5004FBD8-C5C2-DE11-8848-001D09F2841C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4EC2DF19-B7C2-DE11-9C5F-001D09F253C0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4ACA0E5B-EDC2-DE11-BED2-000423D99E46.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4A82C7A3-D4C2-DE11-B41A-003048678098.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4A663065-D0C2-DE11-A9C7-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4A163D75-EFC2-DE11-B17C-000423D9853C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/48C049F0-D3C2-DE11-A00E-001D09F2437B.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/46F394B9-CFC2-DE11-97C9-0030487A1990.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/468EFE98-FAC2-DE11-9409-000423D6B444.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/46376C4F-CEC2-DE11-96E7-0030486780B8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/44920320-D1C2-DE11-BD2C-001617C3B6DE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/42F888E2-BBC2-DE11-AC28-001D09F29597.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/42A24261-EDC2-DE11-9613-000423D944F0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/4256F520-D1C2-DE11-A4C0-000423D98EC8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/40D68156-F5C2-DE11-ACD3-000423D98804.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/403A496E-FDC2-DE11-B458-000423D99F1E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3E8100B1-F5C2-DE11-A768-000423D992A4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3E010A43-E4C2-DE11-A80D-001617E30F48.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3CEB4546-F4C2-DE11-BCBB-000423D98634.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3CEA32B6-CFC2-DE11-A83A-001D09F231C9.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3CC5E22F-DFC2-DE11-A625-000423D944DC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/38D934D8-F7C2-DE11-AC3F-000423D98750.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/38737793-ECC2-DE11-890A-000423D951D4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/364DD314-DDC2-DE11-AAEF-000423D98B5C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/34CAD00D-E9C2-DE11-8C2E-000423D98B6C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3495D6A5-C1C2-DE11-9706-001D09F2841C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/34405CCE-D1C2-DE11-9644-000423D6AF24.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3413DE95-ECC2-DE11-A6B0-000423D94534.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/327A3DCC-DDC2-DE11-A41F-0030487A1FEC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/3250268C-D2C2-DE11-B22D-000423D985E4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/30FB4E03-E9C2-DE11-9C1E-000423D9997E.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/306BA5FA-DAC2-DE11-9155-001617E30F50.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/2ED73B74-0BC3-DE11-988D-000423D98950.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/2E3F5ECE-D1C2-DE11-8277-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/2C2A6D56-E8C2-DE11-9D0A-000423D99AA2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/24B570AE-CDC2-DE11-8A0D-003048D37580.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/24348D10-EEC2-DE11-A5B2-000423D99AA2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/22F6B4DA-EBC2-DE11-BADA-000423D94C68.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/20FEE07D-CBC2-DE11-8027-000423D6BA18.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/2080842F-CCC2-DE11-A825-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/1EF252DF-D8C2-DE11-ADAF-001617E30D52.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/1E90A797-ECC2-DE11-8772-000423D94E1C.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/1AA5B634-DFC2-DE11-A590-000423D996C8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/18EB81E7-F9C2-DE11-9B36-000423D9A2AE.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/18A4D704-E2C2-DE11-B77B-001D09F241B9.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/1875E2CF-D1C2-DE11-9CBE-000423D99BF2.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/162C932F-DFC2-DE11-8D56-000423D94534.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/160F5695-ECC2-DE11-9295-000423D98800.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/16069AB1-DBC2-DE11-AF43-000423D98B08.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/14AB2D4C-CEC2-DE11-BFB4-003048D2C0F0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/148BF98C-D2C2-DE11-B57A-000423D98EC8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/145997CA-DDC2-DE11-81D3-000423D944F0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/128398AC-F5C2-DE11-A296-000423D6CA42.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/124F66F5-CCC2-DE11-867E-003048D2BED6.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/109748FB-DAC2-DE11-88F5-000423D94524.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/106E6C43-EDC2-DE11-8205-000423D94908.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0CDFAF05-BCC2-DE11-A818-000423D99394.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0CDC89FB-DAC2-DE11-B825-000423D998BA.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0C5324B9-CFC2-DE11-B83C-001D09F29538.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0ABCA514-F5C2-DE11-BE62-000423D944DC.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0AA0CF63-E3C2-DE11-A539-000423D98930.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/087186DA-EBC2-DE11-A7F9-000423D9A212.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/080E8524-C5C2-DE11-9194-000423D986A8.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/06BF2310-F0C2-DE11-87C5-000423D987E0.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/04C5C4A3-C1C2-DE11-A82F-0019B9F709A4.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/04AC6AB8-CFC2-DE11-A451-000423D6B444.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/0423630A-C3C2-DE11-B403-001D09F2AD7F.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/00B844A7-E0C2-DE11-8A95-0019B9F72CE5.root',
       '/store/data/Commissioning09/Cosmics/RECO/v9/000/118/367/005A0AA9-E0C2-DE11-BE1B-001D09F2512C.root'] );

secFiles.extend( [
               ] )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50000))

################# Geometry  ######################
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


################# RPC Unpacker  ######################
process.rpcunpacker = cms.EDProducer("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source"),
    doSynchro = cms.bool(False)
)

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

################# RPC Rec Hits  ######################
process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")
process.rpcRecHits.rpcDigiLabel = 'rpcunpacker'

################# DQM Cetral Modules ###################
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'RPC'
process.dqmSaver.convention = 'Online'
############# Message Logger ####################
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('*')
)



############## Output module ##################*_MEtoEDMConverter_*_*
process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('out.root'),
     outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*")
)


process.load("DQMServices.Components.EDMtoMEConverter_cfi")


process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.load("DQM.RPCMonitorClient.RPCTier0Client_cff")
process.load("DQM.RPCMonitorClient.RPCTier0Source_cff")

################# Path ###########################

process.p = cms.Path(process.rpcTier0Source*process.MEtoEDMConverter*process.EDMtoMEConverter*process.rpcTier0Client*process.dqmSaver)
#process.outpath = cms.EndPath(process.out)

                     #*process.EDMtoMEConverter*process.rpcTier0Client*process.dqmSaver)


