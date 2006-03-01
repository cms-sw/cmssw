CREATE OR REPLACE
TRIGGER cond_hv_imon_tg
  BEFORE INSERT ON cond_hv_imon
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_micros('cond_hv_imon', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER cond_hv_i0_tg
  BEFORE INSERT ON cond_hv_i0
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_micros('cond_hv_i0', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER cond_hv_vmon_tg
  BEFORE INSERT ON cond_hv_vmon
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_micros('cond_hv_vmon', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER cond_hv_v0_tg
  BEFORE INSERT ON cond_hv_v0
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_micros('cond_hv_v0', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER cond_hv_t_board_tg
  BEFORE INSERT ON cond_hv_t_board
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_micros('cond_hv_t_board', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/
