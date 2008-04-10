/*
 *  Insert some starting values to the definition tables
 *  These are dummy values for testing purposes.
 */

/* test definitions, pending expert consideration */

INSERT INTO mon_version_def VALUES (mon_version_def_sq.NextVal, 'test01', 'A test version of the monitoring that doesnt really exist');

INSERT INTO mon_run_outcome_def VALUES (mon_run_outcome_def_sq.NextVal, 'success', 'complete success');

@insert_location_data
@insert_run_type_def
