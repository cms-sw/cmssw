/*
 * Create all the data tables referencing lmf_run_iov
 * Requires:  create_laser_core.sql
 */


CREATE TABLE lmf_laser_blue_raw_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_peak		BINARY_FLOAT,
  apd_err		BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_raw_dat ADD CONSTRAINT lmf_laser_blue_raw_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_raw_dat ADD CONSTRAINT lmf_laser_blue_raw_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);

CREATE TABLE lmf_matacq_blue_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  amplitude		BINARY_FLOAT,
  width		        BINARY_FLOAT,
  timeoffset		BINARY_FLOAT
);

ALTER TABLE lmf_matacq_blue_dat ADD CONSTRAINT lmf_matacq_blue_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_matacq_blue_dat ADD CONSTRAINT lmf_matacq_blue_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);

CREATE TABLE lmf_matacq_red_dat (
  iov_id                NUMBER,
  logic_id              NUMBER,
  amplitude             BINARY_FLOAT,
  width                 BINARY_FLOAT,
  timeoffset            BINARY_FLOAT
);
                                                                                                       
ALTER TABLE lmf_matacq_red_dat ADD CONSTRAINT lmf_mataq_red_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_matacq_red_dat ADD CONSTRAINT lmf_matacq_red_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);
                                                                                                       
CREATE TABLE lmf_matacq_green_dat (
  iov_id                NUMBER,
  logic_id              NUMBER,
  amplitude             BINARY_FLOAT,
  width                 BINARY_FLOAT,
  timeoffset            BINARY_FLOAT
);
                                                                                                       
ALTER TABLE lmf_matacq_green_dat ADD CONSTRAINT lmf_mataq_green_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_matacq_green_dat ADD CONSTRAINT lmf_matacq_green_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);
                                                                                                       
CREATE TABLE lmf_matacq_ired_dat (
  iov_id                NUMBER,
  logic_id              NUMBER,
  amplitude             BINARY_FLOAT,
  width                 BINARY_FLOAT,
  timeoffset            BINARY_FLOAT
);
                                                                                                       
ALTER TABLE lmf_matacq_ired_dat ADD CONSTRAINT lmf_mataq_ired_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_matacq_ired_dat ADD CONSTRAINT lmf_matacq_ired_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);
                                                                                                       



CREATE TABLE lmf_laser_ired_raw_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_peak		BINARY_FLOAT,
  apd_err		BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_raw_dat ADD CONSTRAINT lmf_laser_ired_raw_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_raw_dat ADD CONSTRAINT lmf_laser_ired_raw_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_blue_norm_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_over_pnA_mean	BINARY_FLOAT,
  apd_over_pnA_rms	BINARY_FLOAT,
  apd_over_pnB_mean	BINARY_FLOAT,
  apd_over_pnB_rms	BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_norm_dat ADD CONSTRAINT lmf_laser_blue_norm_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_norm_dat ADD CONSTRAINT lmf_laser_blue_norm_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_ired_norm_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_over_pnA_mean	BINARY_FLOAT,
  apd_over_pnA_rms	BINARY_FLOAT,
  apd_over_pnB_mean	BINARY_FLOAT,
  apd_over_pnB_rms	BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_norm_dat ADD CONSTRAINT lmf_laser_ired_norm_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_norm_dat ADD CONSTRAINT lmf_laser_ired_norm_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);


CREATE TABLE lmf_laser_blue_cor_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_over_pnA_mean	BINARY_FLOAT,
  apd_over_pnA_rms	BINARY_FLOAT,
  apd_over_pnB_mean	BINARY_FLOAT,
  apd_over_pnB_rms	BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_cor_dat ADD CONSTRAINT lmf_laser_blue_cor_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_cor_dat ADD CONSTRAINT lmf_laser_blue_cor_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_ired_cor_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  apd_over_pnA_mean	BINARY_FLOAT,
  apd_over_pnA_rms	BINARY_FLOAT,
  apd_over_pnB_mean	BINARY_FLOAT,
  apd_over_pnB_rms	BINARY_FLOAT,
  apd_over_pn_mean	BINARY_FLOAT,
  apd_over_pn_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_cor_dat ADD CONSTRAINT lmf_laser_ired_cor_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_cor_dat ADD CONSTRAINT lmf_laser_ired_cor_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);

CREATE TABLE lmf_laser_blue_time_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  offset        	BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_time_dat ADD CONSTRAINT lmf_laser_blue_time_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_time_dat ADD CONSTRAINT lmf_laser_blue_time_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);

CREATE TABLE lmf_laser_ired_time_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  offset        	BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_time_dat ADD CONSTRAINT lmf_laser_ired_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_time_dat ADD CONSTRAINT lmf_laser_ired_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_blue_coeff_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  xport_coeff	        BINARY_FLOAT,
  xport_coeff_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_coeff_dat ADD CONSTRAINT lmf_laser_blue_coeff_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_coeff_dat ADD CONSTRAINT lmf_laser_blue_coeff_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_ired_coeff_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  xport_coeff	        BINARY_FLOAT,
  xport_coeff_rms	BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_coeff_dat ADD CONSTRAINT lmf_laser_ired_coeff_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_coeff_dat ADD CONSTRAINT lmf_laser_ired_coeff_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_blue_shape_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  alpha			BINARY_FLOAT,
  alpha_rms		BINARY_FLOAT, 
  beta			BINARY_FLOAT,
  beta_rms		BINARY_FLOAT
);

ALTER TABLE lmf_laser_blue_shape_dat ADD CONSTRAINT lmf_laser_blue_shape_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_blue_shape_dat ADD CONSTRAINT lmf_laser_blue_shape_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_laser_ired_shape_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  alpha			BINARY_FLOAT,
  alpha_rms		BINARY_FLOAT, 
  beta			BINARY_FLOAT,
  beta_rms		BINARY_FLOAT
);

ALTER TABLE lmf_laser_ired_shape_dat ADD CONSTRAINT lmf_laser_ired_shape_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_laser_ired_shape_dat ADD CONSTRAINT lmf_laser_ired_shape_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_pn_blue_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  pn_peak		BINARY_FLOAT,
  pn_err		BINARY_FLOAT
);

ALTER TABLE lmf_pn_blue_dat ADD CONSTRAINT lmf_pn_blue_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_pn_blue_dat ADD CONSTRAINT lmf_pn_blue_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_pn_ired_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  pn_peak		BINARY_FLOAT,
  pn_err		BINARY_FLOAT
);

ALTER TABLE lmf_pn_ired_dat ADD CONSTRAINT lmf_pn_ired_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_pn_ired_dat ADD CONSTRAINT lmf_pn_ired_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_pn_test_pulse_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  adc_mean		BINARY_FLOAT,
  adc_rms		BINARY_FLOAT
);

ALTER TABLE lmf_pn_test_pulse_dat ADD CONSTRAINT lmf_pn_test_pulse_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_pn_test_pulse_dat ADD CONSTRAINT lmf_pn_test_pulse_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);



CREATE TABLE lmf_pn_config_dat (
  iov_id		NUMBER,
  logic_id		NUMBER,
  pna_id		NUMBER,
  pnb_id		NUMBER,
  pna_validity		CHAR(1),
  pnb_validity		CHAR(1),
  pnmean_validity	CHAR(1)
);

ALTER TABLE lmf_pn_config_dat ADD CONSTRAINT lmf_pn_config_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE lmf_pn_config_dat ADD CONSTRAINT lmf_pn_config_dat_fk FOREIGN KEY (iov_id) REFERENCES lmf_run_iov (iov_id);
