# Info about Global Tags

Original author: Andrew Brinkerhoff &lt;andrew.wilson.brinkerhoff@cern.ch&gt;

Last edit: February 4, 2020


## L1TMuonEndCapParams

For the L1TMuonEndCapParams tags, here are the appropriate params tags for each year:
- 2016
  - **L1TMuonEndCapParams_static_2016_mc**
    - PtAssignVersion = 5, firmwareVersion = 49999, PhiMatchWindowSt1 = 0
- 2017
  - **L1TMuonEndCapParams_Stage2v1**
    - PtAssignVersion = 7, firmwareVersion = 1497518612, PhiMatchWindowSt1 = 1
    - It's not clear to me why the firmwareVersion for this tag should be 1497518612 and not 1496792995 or 1504018578
    - In the current emulator, makes no difference; but for full consistency 1504018578 might be the best choice for any future "UL" tag for 2017.
- 2018
  - **L1TMuonEndCapParams_Stage2v3_2018_HI_mc**
    - PtAssignVersion = 7, firmwareVersion = 1539271335 (October 11, 2018), PhiMatchWindowSt1 = 1
    - Note that this should be used for *all* 2018 MC, not just Heavy Ion - tag L1TMuonEndCapParams_Stage2v1_2018_mc is WRONG!

## L1TMuonEndCapForest

For the L1TMuonEndCapForest tags, here are the appropriate forest tags for each year:
- 2016
  - **L1TMuonEndCapForest_static_2016_mc**
    - Payload 1d58582f55ae84cf5ec5ea91ebcb8b4a23b1af23, 5.2 MB, loaded 2017-05-24 15:45:10 (UTC)
    - This payload is also included in data tags L1TMuonEndCapForest_Stage2v1_hlt and L1TMuonEndCapForest_Stage2v2_hlt
- 2017 and 2018
  - **L1TMuonEndCapForest_static_Sq_20170613_v7_mc**
  - **L1TMuonEndCapForest_Stage2v1_2018_HI_mc**
    - Payload 821067bddc9f3e5e4e6dd627ecf0c5e453853ccc, 52 MB, loaded 2017-06-13 10:36:43 (UTC)
    - This payload is also included in data tags L1TMuonEndCapForest_Stage2v1_hlt and L1TMuonEndCapForest_Stage2v2_hlt
    - Corresponds to /afs/cern.ch/work/a/abrinke1/public/EMTF/PtAssign2017/XMLs/2017_v7/, which is the latest 2017/2018 XMLs


## How to extract tag and payload information

1. Enter a CMSSW environment and run:
    - `cmsenv`
2. To see the lists of existing EMTF O2O tags, run:
    - `conddb listTags | grep L1TMuonEndCapParams`
    - `conddb listTags | grep L1TMuonEndCapForest`
3. To see the payloads for each tag, do either of the following:
    - Go to: <https://cms-conddb.cern.ch/cmsDbBrowser/list/Prod/tags/YourTagName>
    - Run: `conddb list YourTagName`
4. Using "Payload" from #3 above, run:
    - `conddb dump PayloadHash`

