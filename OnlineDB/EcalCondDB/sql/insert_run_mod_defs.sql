insert into ecal_run_mode_def(run_mode_string) values ('LOCAL');
insert into ecal_run_mode_def(run_mode_string) values ('GLOBAL');
insert into ecal_sequence_type_def(RUN_TYPE_DEF_ID, SEQUENCE_TYPE_STRING) values (116, 'DUMMY_SEQUENCE');
insert into ecal_sequence_type_def(RUN_TYPE_DEF_ID, SEQUENCE_TYPE_STRING) values (116, 'TEST_SEQUENCE');
insert into ecal_sequence_type_def(RUN_TYPE_DEF_ID, SEQUENCE_TYPE_STRING) values (116, 'DEFAULT_SEQUENCE');
insert into ECAL_RUN_CONFIGURATION_DAT (TAG, version, RUN_TYPE_DEF_ID, RUN_MODE_DEF_ID, NUM_OF_SEQUENCES, description) values ('DEFAULT',1,116,1,1,'dummy run');
insert into ecal_sequence_dat (ecal_config_id, SEQUENCE_NUM,NUM_OF_CYCLES,SEQUENCE_TYPE_DEF_ID,description) values (1,1,1,1,'dummy sequence');

