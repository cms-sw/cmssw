<?php
/*
 * db_functions.php
 *
 * All the functions used to connect to and query the DB
 * $Id: db_functions.php,v 1.4 2006/09/27 20:37:57 egeland Exp $
 */

require_once 'common.php';

error_reporting(E_ALL);
ini_set("display_errors","on");

# Globals
$conn = connect();

/* Connects to the database, returns connection handle */
function & connect() {
  $params = get_conn_params();
  $db_user = $params['user'];
  $db_pass = $params['pass'];
  $db_sid  = $params['sid'];

  $conn = oci_connect($db_user, $db_pass, $db_sid);

  if (! $conn ) {
    echo "<h1>ERROR</h1>";
    echo "<p>Connection to DB failed.</p>";
    exit;
  }

  $stmt = oci_parse($conn, "ALTER SESSION SET NLS_DATE_FORMAT='YYYY-MM-DD HH24:MI:SS'");
  oci_execute($stmt);

  return $conn;
}

/* Returns a list of locations available from LOCATION_DEF */
function get_loc_list() {
  global $conn;

  $sql = "SELECT DISTINCT location FROM location_def";
  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);

  return array_values($results['LOCATION']);
}

/* Returns a list of SM available from RUN_DAT */
function get_sm_list() {
  global $conn;

  $sql = "SELECT DISTINCT cv.id1 SM FROM run_dat rdat 
                                 JOIN channelview cv ON rdat.logic_id = cv.logic_id 
                                      AND cv.name = 'EB_supermodule' 
                                      AND cv.name = cv.maps_to";
  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);

  return array_values($results['SM']);
}

/* Returns a list of run types available from RUN_TYPE_DEF */
function get_runtype_list() {
  global $conn;

  $sql = "SELECT DISTINCT run_type FROM run_type_def";
  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);

  return array_values($results['RUN_TYPE']);
}

/* Returns a list of gen_tag available from RUN_TAG */
function get_rungentag_list() {
  global $conn;

  $sql = "SELECT DISTINCT gen_tag FROM run_tag";
  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);

  return array_values($results['GEN_TAG']);
}

/* Returns the min and max run numbers available from RUN_IOV */
function get_run_num_extents() {
  global $conn;

  $sql = "SELECT MIN(riov.run_num) min_run, MAX(riov.run_num) max_run 
           FROM run_iov riov 
           JOIN run_tag rtag ON riov.tag_id = rtag.tag_id
           JOIN run_type_def rdef ON rdef.def_id = rtag.run_type_id
          WHERE rdef.run_type != 'TEST'";
  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  return oci_fetch_assoc($stmt);
}

/* Returns the min and max run_start available from RUN_IOV */
function get_run_date_extents() {
  global $conn;

  $sql = "SELECT MIN(riov.run_start) min_start, MAX(riov.run_start) max_start
           FROM run_iov riov 
           JOIN run_tag rtag ON riov.tag_id = rtag.tag_id
           JOIN run_type_def rdef ON rdef.def_id = rtag.run_type_id
          WHERE rdef.run_type != 'TEST'";

  $stmt = oci_parse($conn, $sql);
  oci_execute($stmt);
  return oci_fetch_assoc($stmt);
}

/* Returns an associative array of headers from the run query.  
   returns array($db_handle => $header) */
function get_runselect_headers() {
  return array('RUN_NUM' => 'Run Number',
	       'SM' => 'SM',
	       'LOCATION' => 'Location',
	       'RUN_TYPE' => 'Run Type',
	       'CONFIG_TAG' => 'Run Config',
	       'CONFIG_VER' => 'Config Ver',
	       'RUN_GEN_TAG' => 'Gen Tag',
	       'RUN_START' => 'Run Start',
	       'RUN_END' => 'Run End',
	       'NUM_EVENTS' => 'Events');
}

/* Returns an associative array of the run query.
  input $params is a set of options from the selection page
  returns array("sql" => The sql of the query, "binds" => a list of bind parameters) */
function build_runselect_sql($params) {
  $selectfrom = "select loc.location, rtype.run_type, rconfig.config_tag, rconfig.config_ver, rtag.gen_tag run_gen_tag,
                        riov.iov_id run_iov_id, riov.run_num, riov.run_start, riov.run_end, riov.db_timestamp run_db_timestamp, 
                        rdat.num_events, cv.id1 SM
                   from location_def loc
                   join run_tag rtag on rtag.location_id = loc.def_id
                   join run_type_def rtype on rtype.def_id = rtag.run_type_id
                   join run_iov riov on riov.tag_id = rtag.tag_id 
                   left join run_dat rdat on rdat.iov_id = riov.iov_id 
                   left join run_config_dat rconfig on rconfig.iov_id = riov.iov_id
                   left join channelview cv on rdat.logic_id = cv.logic_id 
                        and cv.name = 'EB_supermodule'
                        and cv.name = cv.maps_to";
  
  $where = " WHERE loc.location = :location ";

  $binds = array(':location' => $params['location']);

  if ($params['SM'] != 'Any') {
    $where .= " AND (cv.id1 = :SM OR cv.name = 'ECAL')";
    $binds[':SM'] = $params['SM'];
  }

  if ($params['run_type'] == 'All except TEST') {
    $where .= " AND rtype.run_type != 'TEST'";
  } else if ($params['run_type'] != 'All') {
    $where .= " AND rtype.run_type = :run_type ";
    $binds[':run_type'] = $params['run_type'];
  }

  if ($params['run_gen_tag'] != 'All') {
    $where .= " AND rtag.gen_tag = :run_gen_tag ";
    $binds[':run_gen_tag'] = $params['run_gen_tag'];
  }
  
  if ($params['run_select'] == 'run_range') {
    $where .= " AND riov.run_num >= :min_run AND riov.run_num <= :max_run ";
    $binds[':min_run'] = $params['min_run'];
    $binds[':max_run'] = $params['max_run'];
  } elseif ($params['run_select'] == 'date_range') {
    $where .= " AND riov.run_start >= :min_start AND riov.run_start <= :max_start ";
    $binds[':min_start'] = $params['min_start'];
    $binds[':max_start'] = $params['max_start'];
  }

  $run_order = $params['run_order'];
  $orderby = " order by riov.run_num $run_order, rtype.run_type asc, rconfig.config_tag asc, rtag.gen_tag asc";

  $sql = $selectfrom.$where.$orderby;

  if ($params['run_select'] == 'last_100') {
    $sql = "select * from ($sql) where rownum < 101";
  }

  return array('sql' => $sql, 'binds' => $binds);
}

/* Returns an associative array of headers for the monitoring query
   returns array($db_handle => $header) */
function get_monselect_headers() {
  return array('SUBRUN_NUM' => 'Subrun',
	       'GEN_TAG' => 'General Tag',
	       'MON_VER' => 'Monitoring Version',
	       'SUBRUN_START' => 'Subrun Start',
	       'SUBRUN_END' => 'Subrun End',
	       'NUM_EVENTS' => 'Num Events');
}

/* Returns an associative array of the monitoring query.
   input is an IOV_ID from RUN_IOV */
function fetch_mon_data($run_iov_id) {
  global $conn;

  $sql = "select miov.iov_id, miov.subrun_num, miov.subrun_start, miov.subrun_end, mtag.gen_tag, mver.mon_ver,
                 mdat.num_events, mdat.task_list, mdat.task_outcome,
                 dat_exists('MON', miov.iov_id) dat_exists 
            from mon_run_iov miov
            join mon_run_tag mtag on mtag.tag_id = miov.tag_id 
            join mon_version_def mver on mver.def_id = mtag.mon_ver_id
            join mon_run_dat mdat on mdat.iov_id = miov.iov_id
           where miov.run_iov_id = :run_iov_id
           order by miov.subrun_num asc";
  
  $stmt = oci_parse($conn, $sql);

  oci_bind_by_name($stmt, ':run_iov_id', $run_iov_id);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);
  
  return $results;
}

/* Returns an associative array of headers for the DCU query
   returns array($db_handle => $header) */
function get_dcuselect_headers() {
  return array('SINCE' => 'Valid Since',
	       'TILL' => 'Valid Until',
	       'GEN_TAG' => 'General Tag');
}

/* Returns an associative array of the dcu query.
   input is an IOV_ID from RUN_IOV */
function fetch_dcu_data($run_iov_id) {
  global $conn;

  // XXX better check this for correctness of boundries and efficiency later
  $sql = "select diov.iov_id, diov.since, diov.till, dtag.gen_tag,
                 dat_exists('DCU', diov.iov_id) dat_exists
            from dcu_iov diov, dcu_tag dtag, run_iov riov, run_tag rtag
           where diov.tag_id = dtag.tag_id
             and riov.iov_id = :run_iov_id
             and riov.tag_id = rtag.tag_id
             and rtag.location_id = dtag.location_id
             and ((diov.till >= riov.run_start and diov.till <= riov.run_end)
                  or (diov.since >= riov.run_start and diov.till <= riov.run_end)
                  or (diov.since <= riov.run_end and diov.since >= riov.run_start)
                  or (riov.run_start >= diov.since and riov.run_end <= diov.till))
           order by diov.since asc";

  $stmt = oci_parse($conn, $sql);

  oci_bind_by_name($stmt, ':run_iov_id', $run_iov_id);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);
  
  return $results;
}

function get_beamselect_headers() {
  return array('BEAM_FILE' => 'Beam File',
	       'ENERGY' => 'Energy',
	       'PARTICLE' => 'Particle',
	       'SPECIAL_SETTINGS' => 'Special Settings');
}

function fetch_beam_data($run, $loc) {
  global $conn;
  
  if ($loc == 'H4B') {
    $beamtable = "RUN_H4_BEAM_DAT";
    $beamcol = "\"XBH4.BEAM:LAST_FILE_LOADED\"";
  } elseif ($loc == 'H2') {
    $beamtable = "RUN_H2_BEAM_DAT";
    $beamcol = "\"XBH2.BEAM:LAST_FILE_LOADED\"";
  } else {
    return array();
  }

  $sql = "select * from 
            (select rownum R, riov.run_num, bdat.$beamcol beam_file, bdef.energy, bdef.particle, bdef.special_settings
               from ($beamtable bdat
               join run_iov riov on bdat.iov_id = riov.iov_id)
               left outer join beamfile_to_energy_def bdef on bdef.beam_file = bdat.$beamcol
              where riov.run_num <= :run
              order by riov.run_num desc)
           where rownum = 1 ";

  $stmt = oci_parse($conn, $sql);
  
  oci_bind_by_name($stmt, ':run', $run);
  oci_execute($stmt);
  oci_fetch_all($stmt, $results);
  
  return $results;
}

function fetch_field_array($prefix) {
  global $conn;

  $sql = "select table_name, field_name 
           from cond_table_meta t 
           join cond_field_meta f on f.tab_id = t.def_id 
           where t.table_name like :1 and f.is_plottable = 1
           order by table_name, field_name";

  $stmt = oci_parse($conn, $sql);
  
  $like = $prefix.'%';

  oci_bind_by_name($stmt, ':1', $like);
  oci_execute($stmt);

  $results = array();
  while ($row = oci_fetch_array($stmt)) {
    if (!isset($results[$row[0]])) { $results[$row[0]] = array(); }
    array_push($results[$row[0]], $row[0].".".$row[1]);
  }

  return $results;
}

function fetch_plot_data($table, $field, $iov_id) {
  global $conn;

  $sql = "select $field from $table where iov_id = :iov_id";

  $stmt = oci_parse($conn, $sql);
  
  oci_bind_by_name($stmt, ':iov_id', $iov_id);
  oci_execute($stmt);

  oci_fetch_all($stmt, $results);

  return $results;
}

function db_fetch_plot_params($table, $field) {
  global $conn;
  
  $sql = "select t.filled_by, t.content_explanation table_content, t.logic_id_name, t.map_by_logic_id_name, t.logic_id_explanation, 
                f.is_plottable, f.field_type, f.content_explanation field_content, f.label
           from cond_table_meta t join cond_field_meta f on t.def_id = f.tab_id 
          where t.table_name = :1 and f.field_name = :2";
  
  $stmt = oci_parse($conn, $sql);

  oci_bind_by_name($stmt, ':1', $table);
  oci_bind_by_name($stmt, ':2', $field);

  oci_execute($stmt);

  if ($row = oci_fetch_assoc($stmt)) {
    return $row;    
  } else { 
    return 0; 
  }
}


function db_make_rootplot($table, $field, $iov_id, $plottype, $name) {
  global $conn;

  $plot_params = db_fetch_plot_params($table, $field);
  $fmt = "png";
  $title = $plot_params['FIELD_CONTENT'];
  $chan_name = $plot_params['LOGIC_ID_NAME'];
  $map_name = $plot_params['MAP_BY_LOGIC_ID_NAME'];

  if ($plottype == 'histo_all') { 
    $type = 'TH1F'; $grp = 0;
    $sql = "select $field from $table where iov_id = :iov_id";
    $xtitle = $plot_params['LABEL'];
    $ytitle = "";
  } elseif ($plottype == 'histo_grp') {
    $type = 'TH1F'; $grp = 1;
    $sql = "select vd.id1name, cv.id1, d.$field from $table d, channelview cv, viewdescription vd
             where d.iov_id = :iov_id 
               and d.logic_id = cv.logic_id
               and cv.name = cv.maps_to
               and cv.name = vd.name
             order by cv.id1, cv.id2, cv.id3";
    $xtitle = $plot_params['LABEL'];
    $ytitle = "";
  } elseif ($plottype == 'graph_all') {
    $type = 'TGraph'; $grp = 0;
    $sql = "select rownum, $field
              from (select d.$field from $table d, channelview cv
                     where d.iov_id = :iov_id 
                       and d.logic_id = cv.logic_id
                       and cv.name = cv.maps_to
                     order by cv.id1, cv.id2, cv.id3)";
    $xtitle = $plot_params['LOGIC_ID_EXPLANATION'];
    $ytitle = $plot_params['LABEL'];
  } elseif ($plottype == 'graph_grp') { 
    $type = 'TGraph'; $grp = 1;
    $sql = "select id1name, id1, rownum, $field
              from (select vd.id1name, cv.id1, d.$field from $table d, channelview cv, viewdescription vd
                     where d.iov_id = :iov_id 
                       and d.logic_id = cv.logic_id
                       and cv.name = cv.maps_to
                       and cv.name = vd.name
                     order by cv.id1, cv.id2, cv.id3)";
    $xtitle = $plot_params['LOGIC_ID_EXPLANATION'];
    $ytitle = $plot_params['LABEL'];
  } elseif ($plottype == 'map') {
    if (!$map_name) { echo "No map_name.";  return 0; } // Cannot map without an _index type mapping
    $type = 'Map'; $grp = 1;
    $sql = "select id1name, id1, i, j, $field
              from (select vd.id1name, cv.id1, cv.id2 i, cv.id3 j, d.$field from $table d, channelview cv, viewdescription vd
                     where d.iov_id = :iov_id 
                       and d.logic_id = cv.logic_id
                       and cv.name = '$map_name'
                       and cv.maps_to = '$chan_name'
                       and cv.name = vd.name
                     order by cv.id1, cv.id2, cv.id3)";
    echo "SQL:  $sql";
    $xtitle = "";
    $ytitle = "";
  } else { die("Unknown plottype"); }

  $stmt = oci_parse($conn, $sql);
  
  oci_bind_by_name($stmt, ':iov_id', $iov_id);
  oci_execute($stmt);
  
  $n = 0;
  $names = array();
  $rptitle = $title;
  $rpname = $name;
  while ($row = oci_fetch_row($stmt)) {
    // Augment titles and file names if there is grouping
    if ($grp) {
      $grp_name = array_shift($row);
      $curr_grp = array_shift($row);
    }

    // If the group is over add finish the last plot
    if ($n != 0 && $grp && ($last_grp != $curr_grp)) {
      array_push($names, $rpname);
      pclose($rootplot);
    }

    // Open a rootplot handle if it is the first row or the group changed
    if ($n == 0 ||
	($grp && ($last_grp != $curr_grp))) {

      if ($grp) {
	$rptitle = $title." ($grp_name $curr_grp)";
	$rpname = $name.".$grp_name$curr_grp";
      }

      $rootplot = get_rootplot_handle("-T \"$rptitle\" -X \"$xtitle\" -Y \"$ytitle\" $type $fmt $rpname");
      if ( ! $rootplot || get_rootplot_error() ) { return 0; }
    }

    // Write a row of data to the rootplot handle
    $dataline = join(' ', $row)."\n";
    fwrite($rootplot, $dataline);

    // Increment
    $n++;
    if ($grp) { $last_grp = $curr_grp; }
  }
  // Close the last plot
  array_push($names, $rpname);
  pclose($rootplot);
  
  if ($n == 0) {
    echo "N is zero";
  }

  if (get_rootplot_error() || $n == 0) {
    return 0;
  }

  return $names;
}

?>
