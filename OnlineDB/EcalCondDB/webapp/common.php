<?php
/* 
 * common.php
 * 
 * Common variables and functions for the application (basically a config
 * script)
 * $Id: common.php,v 1.11 2007/04/12 11:04:42 fra Exp $
 */

function get_conn_params($location) {
  if ($location && $location == 'P5_MT' ) {
    return array('user' => "cms_ecal",
	       'pass' => "_change_me_",
	       'sid'  => "omds");
  } elseif ( $location && $location == 'P5_Co' ) {
    return array('user' => "cms_ecal_cond",
               'pass' => "_change_me_",
               'sid'  => "omds");
  } else {
    return array('user' => "cond01",
	       'pass' => "_change_me_",
	       'sid'  => "ecalh4db");
  }
}

function get_dqm_url($location, $runtype, $run) {
  if ($location && $location == 'H4B') {
    if ( $run < 18581 ) {
      $url = "http://pctorino1.cern.ch/html/";
    } else {
      $url = "http://pctorino1.cern.ch/html.new/";
    }
  } elseif ($location && $location == 'H2') {
    if ($runtype == 'BEAM') {
      $url = "http://cmshcal04.cern.ch/html/";
    } else {
      $url = "http://pctorino2.cern.ch/html/";
    }
  } elseif ($location && $location == 'P5_MT') {
    $url = "http://lxcms201.cern.ch/mtcc/";
  } elseif ($location && $location == 'P5_Co') {
    $url = "http://ecalod-dqm01/html/";
  } else {
    $url = "http://lxcms201.cern.ch/html/";
  }
  return $url.str_pad($run, 9, '0', STR_PAD_LEFT);
}

function get_dqm_url2($location, $runtype, $run) {
  if ($location && $location == 'H4B') {
    $url = "http://pctorino1.cern.ch/logfile.php?run=" . $run;
  } elseif ($location && $location == 'H2') {
    if ($runtype == 'BEAM') {
      $url = "http://cmshcal04.cern.ch/logfile.php?run=" . $run;
    } else {
      $url = "http://pctorino2.cern.ch/logfile.php?run=" . $run;
      $url = $url . "&runtype=" . $runtype;
    }
  } elseif ($location && $location == 'P5_MT') {
    $url = "log file for run ". $run . "is not available";
  } elseif ($location && $location == 'P5_Co') {
    $url = "http://ecalod-dqm01/logfile.php?run=" . $run;
    $url = $url . "&runtype=" . $runtype;
  } else if ($location && $location == 'H4') {
    $url = "http://lxcms201.cern.ch/logfile.php?run=" . $run;
    $url = $url . "&runtype=" . $runtype;
  }
  return $url;
}

function get_rootplot_path() {
  
}

function get_cache_dir() {

}

function get_datatype_array() {
  return array('Beam' => 'BEAM',
	       'Monitoring' => 'MON',
	       'DCU' => 'DCU',
	       'Laser' => 'LMF',
	       'DCS' => 'DCS');
}

function get_rootplot_handle($args) {
  putenv('ROOTSYS=/afs/cern.ch/cms/external/lcg/external/root/5.12.00/slc3_ia32_gcc323/root');
  putenv('LD_LIBRARY_PATH=/afs/cern.ch/cms/external/lcg/external/root/5.12.00/slc3_ia32_gcc323/root/lib:/afs/cern.ch/cms/sw/slc3_ia32_gcc323/external/boost/1.33.1/lib:$LD_LIBRARY_PATH');
  putenv('ROOTPLOT=CMSSW_0_8_0/bin/slc3_ia32_gcc323/cmsecal_rootplot');

  @system('rm rootplot_error.log');
  $handle = popen("\$ROOTPLOT $args > rootplot_error.log 2>&1", "w") or die('Failed to open rootplot program');

  if (! $handle ) {
    return 0;
  }

  flush();
  fflush($handle);
  if (get_rootplot_error()) {
    pclose($handle);
    return 0;
  }

  return $handle;
}

function get_rootplot_error() {
  $error_file = @fopen('rootplot_error.log', 'r');
  if (! $error_file) { 
    return 0;
  }

  $error_msg = "";
  while ($line = fgets($error_file)) {
    $error_msg .= $line;
  }
  fclose($error_file);
  return $error_msg;
}

function get_task_array() {
  return array('CI' => 'Channel Integrity Task',
	       'CS' => 'Cosmic Task',
	       'LS' => 'Laser Task',
	       'PD' => 'Pedestal Task',
	       'PO' => 'Pedestals Online Task',
	       'TP' => 'Test Pulse Task',
	       'BC' => 'Beam Calo Task',
               'BH' => 'Beam Hodo Task',
               'TT' => 'Trigger Tower Task');
}

function get_task_outcome($list_bits, $outcome_bits) {
  if (!$list_bits && !$outcome_bits) { return false; }
  $tasks = get_task_array();
  
  $result = array();
  foreach(array_keys($tasks) as $i => $taskcode) {
    if ($list_bits & (1 << $i)) {
      $result[$taskcode] = $outcome_bits & (1 << $i);
    }
  }
  return $result;
}

function get_stylelinks( $compact = 0 ) {

  if( $compact == '1' ) {
    return "
<link rel='stylesheet' type='text/css' href='ecalconddb-compact.css'/>
<!--[if IE]>
<link rel='stylesheet' type='text/css' href='fixie.css'/>
<![endif]-->
";
  }
  else {
    return "
<link rel='stylesheet' type='text/css' href='ecalconddb.css'/>
<!--[if IE]>
<link rel='stylesheet' type='text/css' href='fixie.css'/>
<![endif]-->
";
  }
}

?>

