/* 
 * Calls all the SQL scripts needed to create the entire conditions DB
 * Note:  Does not include the CHANNELVIEW and VIEWDESCRIPTION tables/data!
 */

@create_conddb_procedures

@create_run_core
@create_run_data

@create_monitoring_core
@create_monitoring_data

@create_dcu_core
@create_dcu_data

/*
@create_laser_core
@create_laser_data
*/
@create_laser_new

@create_calibration_core
@create_calibration_data

@create_metadata

@insert_defs
@insert_metadata

@dat_exists_func

@show_db
