CREATE SEQUENCE p_con_exec_seq;
CREATE SEQUENCE p_con_payload_seq;
CREATE SEQUENCE p_con_state_seq;

CREATE OR REPLACE TRIGGER p_con_exec_autonumber
BEFORE INSERT ON p_con_execution FOR EACH ROW
BEGIN
    IF :new.exec_id < 0 THEN
        SELECT p_con_exec_seq.nextval INTO :new.exec_id FROM dual;
    END IF;
END;
/

CREATE OR REPLACE TRIGGER p_con_payload_autonumber
BEFORE INSERT ON p_con_execution_payload FOR EACH ROW
BEGIN
    IF :new.exec_id < 0 THEN
        SELECT p_con_exec_seq.currval INTO :new.exec_id FROM dual;
    END IF;

    IF :new.pl_id < 0 THEN
        SELECT p_con_payload_seq.nextval INTO :new.pl_id FROM dual;
    END IF;
END;
/

CREATE OR REPLACE TRIGGER p_con_state_autonumber
BEFORE INSERT ON p_con_payload_state FOR EACH ROW
BEGIN
    IF :new.obj_id <= 0 OR :new.obj_id IS NULL THEN
        SELECT p_con_state_seq.nextval INTO :new.obj_id FROM dual;
    END IF;
END;
/

show errors;
