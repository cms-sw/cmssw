
CREATE TABLE p_con_lock (
	lock_id NUMBER(10),	
	object_name VARCHAR(50)
);

ALTER TABLE p_con_lock ADD constraint object_name_pk primary key (object_name);
	

CREATE TABLE p_con_execution (
	exec_id NUMBER(10), --need a sequence, filled by trigger
	obj_id NUMBER(10) NOT NULL,
	exec_start TIMESTAMP default NULL ,
	exec_end TIMESTAMP default NULL,
	status VARCHAR2(100) default 'FAILED'
);

CREATE TABLE p_con_execution_payload (
	pl_id NUMBER(10), --trigger
	exec_id NUMBER(10),  
	written TIMESTAMP default NULL,
	except_description VARCHAR2(200) DEFAULT 'Not Written'
);


ALTER TABLE p_con_execution ADD constraint p_con_execution_pk primary key (exec_id);
ALTER TABLE p_con_execution ADD constraint p_con_execution_fk foreign key (obj_id) references p_con_payload_state (obj_id);
ALTER TABLE p_con_execution_payload ADD constraint p_con_execution_payload_pk primary key (pl_id);
ALTER TABLE p_con_execution_payload ADD constraint p_con_execution_payload_fk foreign key (exec_id) references p_con_execution (exec_id);
