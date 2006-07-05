/*
 *  Calls all the SQL scripts needed to clear the entire conditions DB
 *  Note:  Does not include the CHANNELVIEW and VIEWDESCRIPTION tables/data!
 */

@clear_calibration_data
@clear_calibration_core
@clear_laser_data
@clear_laser_core
@clear_dcu_data
@clear_dcu_core
@clear_monitoring_data
@clear_monitoring_core
@clear_run_data
@clear_run_core
@clear_conddb_procedures
@clear_metadata

@show_db
