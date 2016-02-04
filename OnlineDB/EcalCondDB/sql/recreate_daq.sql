@clear_daq_config_new
@create_daq_config
@update_tag_and_version
@config_exists_func
@create_syno

grant execute on config_exists to cms_wbm;
grant execute on config_exists to ecal_reader;
@grant_select_on_conf_to_cms_wbm
@grant_select_on_conf_to_ecal_reader

