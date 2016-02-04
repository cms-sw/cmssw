/*
 *  Insert some starting values to the definition tables
 *  These are dummy values for testing purposes.
 */

/* test definitions, pending expert consideration */

INSERT INTO mon_version_def VALUES (mon_version_def_sq.NextVal, 'test01', 'A test version of the monitoring that doesnt really exist');

INSERT INTO mon_run_outcome_def VALUES (mon_run_outcome_def_sq.NextVal, 'success', 'complete success');

@insert_location_data
@insert_run_type_def

INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'CI',0, 'Channel Integrity', 'Channel Integrity Analysis'); 
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'CS',1, 'Cosmic', 'Cosmic Run Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'LS',2, 'Laser', 'Laser Run Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'PD',3, 'Pedestals', 'Pedestal Run Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'PO',4, 'Pedestals Online', 'Pedestal Analysis from the pre-samples on Any Run');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'TP',5, 'Test Pulse','Test Pulse Run Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'BC',6, 'Beam Calo','Beam Run ECAL Analysis (Test-Beam only)');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'BH',7, 'Beam Hodo','Beam Run Hodoscope Analysis (Test-beam only)'); 
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'TT',8, 'TriggerTower', 'Trigger Tower Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'CL',9, 'Cluster', 'Cluster Analysis on Beam Runs'); 
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'TM',10,'Timing', 'Timing Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'LD',11,'Led','Led Run Analysis (EE)');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'SF',12,'Status Flags','Status Flags Analysis');
INSERT INTO mon_task_def VALUES (mon_task_def_sq.NextVal, 'OC',13,'Occupancy','Occupancy Analysis on Beam Runs');
