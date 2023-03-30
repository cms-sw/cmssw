Unit test: `testHLTFiltersDQMonitor`
------------------------------------

Test of the DQM plugin `HLTFiltersDQMonitor`.

  - To run the test via `scram`
    ```sh
    scram build runtests_testHLTFiltersDQMonitor
    ```

  - To run the test without `scram`
    ```sh
    LOCALTOP="${CMSSW_BASE}" "${CMSSW_BASE}"/src/DQMOffline/Trigger/test/testHLTFiltersDQMonitor.sh
    ```

  - To show info on command-line arguments of `testHLTFiltersDQMonitor_cfg.py`
    ```sh
    python3 "${CMSSW_BASE}"/src/DQMOffline/Trigger/test/testHLTFiltersDQMonitor_cfg.py -h
    ```

  - To execute cmsRun with `testHLTFiltersDQMonitor_cfg.py` (example)
    ```sh
    cmsRun "${CMSSW_BASE}"/src/DQMOffline/Trigger/test/testHLTFiltersDQMonitor_cfg.py -- -t 4 -s 0 -o tmp.root -n 100
    ```

  - To create a bare ROOT file from the DQMIO output of `testHLTFiltersDQMonitor_cfg.py`,
    run the harvesting step as follows
    ```sh
    cmsRun "${CMSSW_BASE}"/src/DQMOffline/Trigger/test/harvesting_cfg.py -- -i file:tmp.root
    ```
