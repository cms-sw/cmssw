This directory contains utilities to create a spreadsheet summarising the content of a given HLT menu.
Such a spreadsheet is normally produced by TSG upon the release of a new HLT menu to be used
for data-taking and/or central production of Monte-Carlo samples.

 - The JSON file `hltPathOwners.json` contains
   the latest information on the triggers (Paths) in the HLT combined table.
   For every given trigger, this file lists the groups which are responsible for it ("owners"),
   and whether or not this trigger is to be included in the "online" version of the HLT menu
   (as opposed to being included only in the "offline" menus used for MC productions).

 - `hltListPathsWithoutOwners`:
   this scripts print to stdout the names of the triggers
   of a given HLT configuration which do not have a owner,
   based on the content of the JSON file provided as input to the script.

 - `hltMenuContentToCSVs`:
   this scripts creates files in CSV format which summarise the content of a given HLT configuration.
   Each of these CSV files corresponds to one of the tabs of
   the spreadsheet produced by TSG upon the release of a new HLT menu.

Instructions for creating a spreadsheet for the release of a HLT menu.

 1. Check if there are Paths without owners in the target configuration.
    ```
    ./hltListPathsWithoutOwners /dev/CMSSW_13_2_0/GRun --meta hltPathOwners.json
    ```

 2. If there are Paths without owners, update the JSON file accordingly.
      - Update `hltPathOwners.json` manually.
      - Create a new version of it with proper formatting as `tmp.json` by using the commands below in `python3`.
        ```python
        import json
        json.dump(json.load(open('hltPathOwners.json')),open('tmp.json','w'),sort_keys=True, indent=2)
        ```
      - Check the content of `tmp.json`. If correct, rename it manually to `hltPathOwners.json`.
        ```bash
        mv tmp.json hltPathOwners.json
        ```

 3. Create the `.csv` files summarising the content of the HLT menu.
    ```bash
    ./hltMenuContentToCSVs /dev/CMSSW_13_2_0/GRun --meta hltPathOwners.json --prescale 2e34
    ```
    In the command above, provide as argument to `--prescale` the name of the PS column
    to be considered as the main/default PS column for that HLT menu.

 4. Copy one of the previous HLT-menu spreadsheets, or copy
    [this template](https://docs.google.com/spreadsheets/d/11Jubd_1Mgh9bueaQUH4Clc-SpQqZ7q1LaUT_QJa9NOQ).
    This includes the "conditional formatting" used to color each slot based on its content.

 5. The spreadsheet contains four tabs.
    The first one ("metadata") is to be filled manually.
    Each of the other three tabs corresponds to one of the CSV files created by `hltMenuContentToCSVs`.
    For each of those tabs, first ensure it be empty (ctrl-a + delete),
    and then fill it by importing the corresponding CSV file
    (the name of the tab is similar to the name of the corresponding CSV file).
    When importing one of these files, select `|` as separator for the CSV lines.
