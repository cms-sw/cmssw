

CREATE OR REPLACE function test_update_tag_and_version
( cndc_table IN VARCHAR2,
  tag IN varchar2,
  version IN integer)  return INTEGER IS
 /* 
 *
 * Procedure to attribute a new version number if a tag exists with that name
 */
  sql_str VARCHAR(1000);
  num_tags INTEGER;
  last_version INTEGER;
  new_version INTEGER;
  cur_version INTEGER;	

  BEGIN

  cur_version := version;
	IF version IS NULL THEN 
	new_version :=0;
	cur_version :=0;
	END IF;

    -- Make sure that this tag does not exist
    sql_str := 'SELECT count(*) FROM ' || cndc_table || ' WHERE tag = :s ';
    EXECUTE IMMEDIATE sql_str INTO num_tags USING tag ;

    IF num_tags = 0 THEN
	new_version := cur_version;
    END IF;	

    -- Make sure that if it exists the proposed tag is higher 

    sql_str := 'SELECT max(version) FROM ' || cndc_table || ' WHERE tag = :s ';
    EXECUTE IMMEDIATE sql_str INTO last_version USING tag ;

    IF last_version IS NOT NULL THEN
	IF last_version>=cur_version THEN
          new_version:= last_version+1;
	ELSIF 1=1 THEN
          new_version:= cur_version;     
        END IF;
    END IF;

    return new_version;
end;
/
show errors;



CREATE OR REPLACE TRIGGER ecal_configuration_dat_ver_tg
  BEFORE INSERT ON ecal_run_configuration_dat
  FOR EACH ROW
    begin
  select test_update_tag_and_version('ECAL_RUN_CONFIGURATION_DAT', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;

