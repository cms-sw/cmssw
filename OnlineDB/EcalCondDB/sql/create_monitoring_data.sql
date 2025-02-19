/*
 *  Creates all the data tables referencing mon_run_iov
 *  Requires:  create_mon_core.sql
 */



CREATE TABLE mon_run_outcome_def (
  def_id		NUMBER(10),
  short_desc		VARCHAR2(100),
  long_desc		VARCHAR2(1000)
);

ALTER TABLE mon_run_outcome_def ADD CONSTRAINT mon_run_outcome_def_pk PRIMARY KEY (def_id);
ALTER TABLE mon_run_outcome_def ADD CONSTRAINT mon_run_outcome_def_uk UNIQUE (short_desc);

CREATE SEQUENCE mon_run_outcome_def_sq INCREMENT BY 1 START WITH 1;

CREATE TABLE mon_run_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- ECAL
  num_events		NUMBER(10),
  run_outcome_id	NUMBER(10),
  rootfile_name         VARCHAR2(100),
  task_list		NUMBER(10),
  task_outcome		NUMBER(10)
);

ALTER TABLE mon_run_dat ADD CONSTRAINT mon_run_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_run_dat ADD CONSTRAINT mon_run_dat_fk1 FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);
ALTER TABLE mon_run_dat ADD CONSTRAINT mon_run_dat_fk2 FOREIGN KEY (run_outcome_id) REFERENCES mon_run_outcome_def (def_id);



CREATE TABLE mon_crystal_status_def (
  def_id		NUMBER(10),
  short_desc		VARCHAR2(100),
  long_desc		VARCHAR2(1000)
);

ALTER TABLE mon_crystal_status_def ADD CONSTRAINT mon_crystal_status_def_pk PRIMARY KEY (def_id);
ALTER TABLE mon_crystal_status_def ADD CONSTRAINT mon_crystal_status_def_uk UNIQUE (short_desc);

CREATE SEQUENCE mon_crystal_status_def_sq INCREMENT BY 1 START WITH 1;

CREATE TABLE mon_crystal_status_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  status_g1		NUMBER(10),
  status_g6		NUMBER(10),
  status_g12		NUMBER(10)
);

ALTER TABLE mon_crystal_status_dat ADD CONSTRAINT mon_crystal_status_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_crystal_status_dat ADD CONSTRAINT mon_crystal_status_dat_fk1 FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);
ALTER TABLE mon_crystal_status_dat ADD CONSTRAINT mon_crystal_status_dat_fk2 FOREIGN KEY (status_g1) REFERENCES mon_crystal_status_def (def_id);
ALTER TABLE mon_crystal_status_dat ADD CONSTRAINT mon_crystal_status_dat_fk3 FOREIGN KEY (status_g6) REFERENCES mon_crystal_status_def (def_id);
ALTER TABLE mon_crystal_status_dat ADD CONSTRAINT mon_crystal_status_dat_fk4 FOREIGN KEY (status_g12) REFERENCES mon_crystal_status_def (def_id);



CREATE TABLE mon_pn_status_def (
  def_id		NUMBER(10),
  short_desc		VARCHAR2(100),
  long_desc		VARCHAR2(1000)
);

ALTER TABLE mon_pn_status_def ADD CONSTRAINT mon_pn_status_def_pk PRIMARY KEY (def_id);
ALTER TABLE mon_pn_status_def ADD CONSTRAINT mon_pn_status_def_uk UNIQUE (short_desc);

CREATE SEQUENCE mon_pn_status_def_sq INCREMENT BY 1 START WITH 1;

CREATE TABLE mon_pn_status_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  status_g1		NUMBER(10),
  status_g16		NUMBER(10)
);

ALTER TABLE mon_pn_status_dat ADD CONSTRAINT mon_pn_status_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_status_dat ADD CONSTRAINT mon_pn_status_dat_fk1 FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);
ALTER TABLE mon_pn_status_dat ADD CONSTRAINT mon_pn_status_dat_fk2 FOREIGN KEY (status_g1) REFERENCES mon_pn_status_def (def_id);
ALTER TABLE mon_pn_status_dat ADD CONSTRAINT mon_pn_status_dat_fk3 FOREIGN KEY (status_g16) REFERENCES mon_pn_status_def (def_id);



CREATE TABLE mon_crystal_consistency_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- cystal
  processed_events	NUMBER(10),
  problematic_events	NUMBER(10),
  problems_id		NUMBER(10),
  problems_gain_zero	NUMBER(10),
  problems_gain_switch	NUMBER(10),
  task_status		CHAR(1)
);

ALTER TABLE mon_crystal_consistency_dat ADD CONSTRAINT mon_crystal_consistency_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_crystal_consistency_dat ADD CONSTRAINT mon_crystal_consistency_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_tt_consistency_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- TT
  processed_events	NUMBER(10),
  problematic_events	NUMBER(10),
  problems_id		NUMBER(10),
  problems_size		NUMBER(10),
  problems_LV1		NUMBER(10),
  problems_bunch_X	NUMBER(10),
  task_status		CHAR(1)
);

ALTER TABLE mon_tt_consistency_dat ADD CONSTRAINT mon_tt_consistency_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_tt_consistency_dat ADD CONSTRAINT mon_tt_consistency_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);


CREATE TABLE mon_occupancy_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  events_over_low_threshold	NUMBER(10),
  events_over_high_threshold	NUMBER(10),
  avg_energy		BINARY_FLOAT
);

ALTER TABLE mon_occupancy_dat ADD CONSTRAINT mon_occupancy_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_occupancy_dat ADD CONSTRAINT mon_occupancy_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pedestals_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  ped_mean_g1		BINARY_FLOAT, 
  ped_rms_g1		BINARY_FLOAT, 
  ped_mean_g6		BINARY_FLOAT, 
  ped_rms_g6		BINARY_FLOAT, 
  ped_mean_g12		BINARY_FLOAT,
  ped_rms_g12		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pedestals_dat ADD CONSTRAINT mon_pedestals_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pedestals_dat ADD CONSTRAINT mon_pedestals_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pedestals_online_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  adc_mean_g12		BINARY_FLOAT,
  adc_rms_g12		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pedestals_online_dat ADD CONSTRAINT mon_pedestals_online_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pedestals_online_dat ADD CONSTRAINT mon_pedestals_online_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pedestal_offsets_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- TT
  dac_g1		NUMBER(10),
  dac_g6		NUMBER(10),
  dac_g12		NUMBER(10),
  task_status		CHAR(1)
);

ALTER TABLE mon_pedestal_offsets_dat ADD CONSTRAINT mon_pedestal_offsets_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pedestal_offsets_dat ADD CONSTRAINT mon_pedestal_offsets_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_test_pulse_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  adc_mean_g1		BINARY_FLOAT, 
  adc_mean_g6		BINARY_FLOAT, 
  adc_mean_g12		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT, 
  adc_rms_g6		BINARY_FLOAT, 
  adc_rms_g12		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_test_pulse_dat ADD CONSTRAINT mon_test_pulse_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_test_pulse_dat ADD CONSTRAINT mon_test_pulse_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pulse_shape_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  g1_avg_sample_01	BINARY_FLOAT,
  g1_avg_sample_02	BINARY_FLOAT,
  g1_avg_sample_03	BINARY_FLOAT,
  g1_avg_sample_04	BINARY_FLOAT,
  g1_avg_sample_05	BINARY_FLOAT,
  g1_avg_sample_06	BINARY_FLOAT,
  g1_avg_sample_07	BINARY_FLOAT,
  g1_avg_sample_08	BINARY_FLOAT,
  g1_avg_sample_09	BINARY_FLOAT,
  g1_avg_sample_10	BINARY_FLOAT,
  g6_avg_sample_01	BINARY_FLOAT,
  g6_avg_sample_02	BINARY_FLOAT,
  g6_avg_sample_03	BINARY_FLOAT,
  g6_avg_sample_04	BINARY_FLOAT,
  g6_avg_sample_05	BINARY_FLOAT,
  g6_avg_sample_06	BINARY_FLOAT,
  g6_avg_sample_07	BINARY_FLOAT,
  g6_avg_sample_08	BINARY_FLOAT,
  g6_avg_sample_09	BINARY_FLOAT,
  g6_avg_sample_10	BINARY_FLOAT,
  g12_avg_sample_01	BINARY_FLOAT,
  g12_avg_sample_02	BINARY_FLOAT,
  g12_avg_sample_03	BINARY_FLOAT,
  g12_avg_sample_04	BINARY_FLOAT,
  g12_avg_sample_05	BINARY_FLOAT,
  g12_avg_sample_06	BINARY_FLOAT,
  g12_avg_sample_07	BINARY_FLOAT,
  g12_avg_sample_08	BINARY_FLOAT,
  g12_avg_sample_09	BINARY_FLOAT,
  g12_avg_sample_10	BINARY_FLOAT
);

ALTER TABLE mon_pulse_shape_dat ADD CONSTRAINT mon_pulse_shape_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pulse_shape_dat ADD CONSTRAINT mon_pulse_shape_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_shape_quality_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  avg_chi2		BINARY_FLOAT
);

ALTER TABLE mon_shape_quality_dat ADD CONSTRAINT mon_shape_quality_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_shape_quality_dat ADD CONSTRAINT mon_shape_quality_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_delays_tt_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- TT
  delay_mean		BINARY_FLOAT,
  delay_rms		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_delays_tt_dat ADD CONSTRAINT mon_delays_tt_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_delays_tt_dat ADD CONSTRAINT mon_delays_tt_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_mem_ch_consistency_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- EB_mem_channel
  processed_events	NUMBER(10),
  problematic_events	NUMBER(10),
  problems_id		NUMBER(10),
  problems_gain_zero	NUMBER(10),
  problems_gain_switch	NUMBER(10),
  task_status		CHAR(1)
);

ALTER TABLE mon_mem_ch_consistency_dat ADD CONSTRAINT mon_mem_ch_consistency_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_mem_ch_consistency_dat ADD CONSTRAINT mon_mem_ch_consistency_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_mem_tt_consistency_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- EB_mem_tt
  processed_events	NUMBER(10),
  problematic_events	NUMBER(10),
  problems_id		NUMBER(10),
  problems_size		NUMBER(10),
  problems_LV1		NUMBER(10),
  problems_bunch_X	NUMBER(10),
  task_status		CHAR(1)
);

ALTER TABLE mon_mem_tt_consistency_dat ADD CONSTRAINT mon_mem_tt_consistency_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_mem_tt_consistency_dat ADD CONSTRAINT mon_mem_tt_consistency_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_blue_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_blue_dat ADD CONSTRAINT mon_pn_blue_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_blue_dat ADD CONSTRAINT mon_pn_blue_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_green_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_green_dat ADD CONSTRAINT mon_pn_green_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_green_dat ADD CONSTRAINT mon_pn_green_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_red_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_red_dat ADD CONSTRAINT mon_pn_red_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_red_dat ADD CONSTRAINT mon_pn_red_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_ired_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_ired_dat ADD CONSTRAINT mon_pn_ired_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_ired_dat ADD CONSTRAINT mon_pn_ired_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_ped_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_ped_dat ADD CONSTRAINT mon_pn_ped_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_ped_dat ADD CONSTRAINT mon_pn_ped_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id); 



CREATE TABLE mon_pn_mgpa_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_mgpa_dat ADD CONSTRAINT mon_pn_mgpa_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_mgpa_dat ADD CONSTRAINT mon_pn_mgpa_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_blue_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  apd_mean		BINARY_FLOAT,
  apd_rms		BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_laser_blue_dat ADD CONSTRAINT mon_laser_blue_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_blue_dat ADD CONSTRAINT mon_laser_blue_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_green_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  apd_mean		BINARY_FLOAT,
  apd_rms		BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_laser_green_dat ADD CONSTRAINT mon_laser_green_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_green_dat ADD CONSTRAINT mon_laser_green_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_red_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  apd_mean		BINARY_FLOAT,
  apd_rms		BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_laser_red_dat ADD CONSTRAINT mon_laser_red_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_red_dat ADD CONSTRAINT mon_laser_red_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_ired_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  apd_mean		BINARY_FLOAT,
  apd_rms		BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_laser_ired_dat ADD CONSTRAINT mon_laser_ired_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_ired_dat ADD CONSTRAINT mon_laser_ired_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_h4_table_position_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- SM
  table_x		BINARY_FLOAT,
  table_y		BINARY_FLOAT
);

ALTER TABLE mon_h4_table_position_dat ADD CONSTRAINT mon_h4_table_position_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_h4_table_position_dat ADD CONSTRAINT mon_h4_table_position_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_status_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- SM
  laser_power		BINARY_FLOAT,
  laser_filter		BINARY_FLOAT,
  laser_wavelength	BINARY_FLOAT,
  laser_fanout		CHAR(1)
);

ALTER TABLE mon_laser_status_dat ADD CONSTRAINT mon_laser_status_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_status_dat ADD CONSTRAINT mon_laser_status_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_laser_pulse_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- SM
  pulse_height_mean	BINARY_FLOAT,
  pulse_height_rms	BINARY_FLOAT,
  pulse_width_mean	BINARY_FLOAT,
  pulse_width_rms	BINARY_FLOAT
);

ALTER TABLE mon_laser_pulse_dat ADD CONSTRAINT mon_laser_pulse_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_laser_pulse_dat ADD CONSTRAINT mon_laser_pulse_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);


CREATE TABLE mon_timing_crystal_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_crystal_dat ADD CONSTRAINT mon_timing_crystal_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_crystal_dat ADD CONSTRAINT mon_timing_crystal_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_tt_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- tt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_tt_dat ADD CONSTRAINT mon_timing_tt_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_tt_dat ADD CONSTRAINT mon_timing_tt_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);


CREATE TABLE mon_LED1_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  vpt_mean		BINARY_FLOAT,
  vpt_rms		BINARY_FLOAT,
  vpt_over_pn_mean	BINARY_FLOAT,
  vpt_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_led1_dat ADD CONSTRAINT mon_led1_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_led1_dat ADD CONSTRAINT mon_led1_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);


CREATE TABLE mon_LED2_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- crystal
  vpt_mean		BINARY_FLOAT,
  vpt_rms		BINARY_FLOAT,
  vpt_over_pn_mean	BINARY_FLOAT,
  vpt_over_pn_rms	BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_led2_dat ADD CONSTRAINT mon_led2_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_led2_dat ADD CONSTRAINT mon_led2_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_led1_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_led1_dat ADD CONSTRAINT mon_pn_led1_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_led1_dat ADD CONSTRAINT mon_pn_led1_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_pn_led2_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- PN diode
  adc_mean_g1		BINARY_FLOAT,
  adc_rms_g1		BINARY_FLOAT,
  adc_mean_g16		BINARY_FLOAT,
  adc_rms_g16		BINARY_FLOAT,
  ped_mean_g1		BINARY_FLOAT,
  ped_rms_g1		BINARY_FLOAT,
  ped_mean_g16		BINARY_FLOAT,
  ped_rms_g16		BINARY_FLOAT,
  task_status		CHAR(1)
);

ALTER TABLE mon_pn_led2_dat ADD CONSTRAINT mon_pn_led2_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_pn_led2_dat ADD CONSTRAINT mon_pn_led2_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);


CREATE TABLE mon_timing_xtal_lb_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_lb_dat ADD CONSTRAINT mon_timing_xtal_lb_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_lb_dat ADD CONSTRAINT mon_timing_xtal_lb_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_xtal_lg_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_lg_dat ADD CONSTRAINT mon_timing_xtal_lg_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_lg_dat ADD CONSTRAINT mon_timing_xtal_lg_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_xtal_lr_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_lr_dat ADD CONSTRAINT mon_timing_xtal_lr_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_lr_dat ADD CONSTRAINT mon_timing_xtal_lr_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_xtal_li_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_li_dat ADD CONSTRAINT mon_timing_xtal_li_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_li_dat ADD CONSTRAINT mon_timing_xtal_li_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_xtal_l1_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_l1_dat ADD CONSTRAINT mon_timing_xtal_l1_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_l1_dat ADD CONSTRAINT mon_timing_xtal_l1_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_xtal_l2_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_xtal_l2_dat ADD CONSTRAINT mon_timing_xtal_l2_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_xtal_l2_dat ADD CONSTRAINT mon_timing_xtal_l2_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



CREATE TABLE mon_timing_TT_lb_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_lb_dat ADD CONSTRAINT mon_timing_TT_lb_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_lb_dat ADD CONSTRAINT mon_timing_TT_lb_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_TT_lg_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_lg_dat ADD CONSTRAINT mon_timing_TT_lg_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_lg_dat ADD CONSTRAINT mon_timing_TT_lg_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_TT_lr_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_lr_dat ADD CONSTRAINT mon_timing_TT_lr_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_lr_dat ADD CONSTRAINT mon_timing_TT_lr_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_TT_li_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_li_dat ADD CONSTRAINT mon_timing_TT_li_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_li_dat ADD CONSTRAINT mon_timing_TT_li_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_TT_l1_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_l1_dat ADD CONSTRAINT mon_timing_TT_l1_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_l1_dat ADD CONSTRAINT mon_timing_TT_l1_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);

CREATE TABLE mon_timing_TT_l2_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- xt
  timing_mean		BINARY_FLOAT,
  timing_rms		BINARY_FLOAT,
  task_status           CHAR(1)
);

ALTER TABLE mon_timing_TT_l2_dat ADD CONSTRAINT mon_timing_TT_l2_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE mon_timing_TT_l2_dat ADD CONSTRAINT mon_timing_TT_l2_dat_fk FOREIGN KEY (iov_id) REFERENCES mon_run_iov (iov_id);



