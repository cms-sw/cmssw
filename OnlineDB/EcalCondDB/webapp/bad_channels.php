<?php

require_once 'common.php';
require_once 'db_functions.php';

$tasks = get_task_array();

$loc      = $_GET['loc'];
$taskcode = $_GET['taskcode'];
$run      = $_GET['run'];
$iov_id   = $_GET['iov_id'];

$conn = connect($_GET['loc']);

?>

<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>
<?php
echo "<title>bad channels list - run ".$run." @ ".$loc."</title>";
echo get_stylelinks();
?>

</head>
<body>

<?php

// Tables to output for each task code
$tables['CI'][0] = "MON_CRYSTAL_CONSISTENCY_DAT";
$tables['CI'][1] = "MON_TT_CONSISTENCY_DAT";
$tables['CI'][2] = "MON_MEM_CH_CONSISTENCY_DAT";
$tables['CI'][3] = "MON_MEM_TT_CONSISTENCY_DAT";
  
$tables['PO'][0] = "MON_PEDESTALS_ONLINE_DAT";
$tables['PD'][0] = "MON_PEDESTALS_DAT";
$tables['PD'][1] = "MON_PN_PED_DAT";
  
$tables['TP'][0] = "MON_TEST_PULSE_DAT";
$tables['TP'][1] = "MON_PN_MGPA_DAT";
  
$tables['LS'][0] = "MON_LASER_RED_DAT";
$tables['LS'][1] = "MON_PN_RED_DAT";
$tables['LS'][2] = "MON_LASER_BLUE_DAT";
$tables['LS'][3] = "MON_PN_BLUE_DAT";
$tables['LS'][4] = "MON_LASER_GREEN_DAT";
$tables['LS'][5] = "MON_PN_GREEN_DAT";
$tables['LS'][6] = "MON_LASER_IRED_DAT";
$tables['LS'][7] = "MON_PN_IRED_DAT";
  
// Page Title
echo "<center>";
echo "<h1>List of bad channels - run ".$run." @ ".$loc."</h1>";
echo "</center>";
echo "<hr/>";
echo "<h2>".$tasks[$taskcode]."</h2>";
echo "<hr/>";
  
if ( $taskcode == 'CS' || $taskcode == 'BC' || $taskcode == 'BH' || $taskcode == 'TT' ) {
  echo "<h3>No bad channel lists are produced by this task</h3>";
  echo "<hr/>";
  exit;
}
  
foreach ($tables[$taskcode] as $table) {
  $data = fetch_mon_dataset_data($table, $iov_id, " task_status = :ts", array(':ts' => "0"));
  $headers = fetch_mon_dataset_headers($table);
  $headers = reorder_columns($table, $headers);
  $nrows = count($data['RUN']);
  $ncols = count($headers);
    
  // Table Title
  echo "<h4>$table<h4>";
    
  if ($nrows == 0) {
    echo "<h5>No bad channels in $table</h5>";
    echo "<hr/>";
    continue;
  }
    
  // Header row
  echo "<table class='bads'>";
  echo "<tr>";
  foreach($headers as $code => $head) {
    echo "<th>$head</th>";
    if ($head == 'crystal_number') {
      add_xtal_headers();
    }
  }
  echo "</tr>";
    
  // Data rows
  for ($i = 0; $i < $nrows; $i++) {
    if ($i % 2 == 0) { echo "<tr>"; }
    else { echo "<tr class='bads'>"; }
      
    foreach($headers as $code => $head) {
      $field = $data[$code][$i];
      // Useing a regexp to find floats
      // Somehow is_float() doesn't work for Oracle formatted floats...
      if (preg_match("/\d+E[\+\-]\d+/", $field)) {
	$field = sprintf("%6.1f", $field);
      }
      echo "<td>".$field."</td>";
      if ($head == 'crystal_number') {
	add_xtal_columns($field);
      }
    }
    echo "</tr>";
  }
    
  echo "</table>";
  echo "<hr/>";
}


function add_xtal_headers() {
  echo "<th>eta</th><th>phi</th><th>tower</th><th>strip</th><th>crystal in strip</th>";
}

function add_xtal_columns($xtal) {
  $eta = 1 + floor(($xtal -1)/20);
  $phi = 1 + ($xtal -1)%20;
  $tower = 1 + 4*floor(($eta -1)/5) + floor(($phi -1)/5);

  $nxtal = 1;
  $strip = 1;

  if ( ($tower > 12 && $tower < 21 ) || ($tower > 28 && $tower < 37) ||
       ($tower > 44 && $tower < 53 ) || ($tower > 60 && $tower < 69) ) {
    $cryInTower = (($eta -1)%5)*5;
    if ( (($eta -1)%5)%2 == 0 ) {
      $cryInTower += ($phi -1)%5;
    } else{
      $cryInTower += 5 - 1 - ($phi -1)%5;
    }
  } else {
    $cryInTower = (5 - 1 - ($eta -1)%5)*5;
    if ( (($eta-1)%5)%2 == 0 ) {
      $cryInTower += 5 - 1 - ($phi -1)%5;
    } else{
      $cryInTower += ($phi -1)%5;
    }
  }

  $nxtal = $cryInTower%5 + 1;
  $strip = floor($cryInTower/5) + 1;

  echo "<td>$eta</td><td>$phi</td><td>$tower</td><td>$strip</td><td>$nxtal</td>";
}

function reorder_columns($table, $col_headers) {
  $order = array_keys($col_headers);

  // Preferred column order based on table name
  if ($table == 'MON_PEDESTALS_DAT') {
    $order = array('RUN', 'ID1', 'ID2', 
		   'PED_MEAN_G12', 'PED_RMS_G12', 
		   'PED_MEAN_G6', 'PED_RMS_G6', 
		   'PED_MEAN_G1', 'PED_RMS_G1',
		   'TASK_STATUS');
  }
  
  $new_headers = array();
  foreach ($order as $code) {
    $new_headers[$code] = $col_headers[$code];
  }

  return ($new_headers);
}

?>

</body>
</html>
