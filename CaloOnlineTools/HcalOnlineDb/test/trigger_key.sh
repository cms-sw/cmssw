#!/bin/bash
./xmlToolsRun --create-trigger-key

cat test_config_data/trigger_keys.xml | sed s/test_trigger_key_id_value/`grep -m 1 'TRIGGER_KEY_ID' HCAL_trigger_key.xml | sed 's/\ \+<TRIGGER_KEY_ID>\(.\+\)<\/TRIGGER_KEY_ID>*/\1/g'`/ > trigger_keys.xml

zip -r -j trigger_key.zip trigger_keys.xml
zip -r -j trigger_key.zip HCAL_trigger_key.xml
