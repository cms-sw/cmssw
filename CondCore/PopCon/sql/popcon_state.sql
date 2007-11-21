CREATE TABLE p_con_payload_state (
	obj_id NUMBER(10) NOT NULL,
	name VARCHAR2(100) NOT NULL,
	payload_size NUMBER(10) DEFAULT 0,
	except_description VARCHAR2(200) DEFAULT NULL,
	manual_override VARCHAR2(100) DEFAULT NULL,
	connect_string VARCHAR2(200) DEFAULT NULL
);

ALTER TABLE p_con_payload_state ADD constraint p_con_payload_state_pk primary key (obj_id);
