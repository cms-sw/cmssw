<?php
/*
 * list_runs.php
 *
 * List a page of runs and subtables for various data categories
 * $Id: list_runs.php,v 1.7 2007/04/12 11:04:42 fra Exp $
 */

require_once 'common.php';
require_once 'db_functions.php';
require_once 'pager_functions.php';

$conn = connect($_GET['location']);
/* $compact = isset($_GET['compact']); */
$compact = 1 - isset($_GET['expanded']);

function input_errors() {
  $error = "";

  foreach (array('location', 'run_type', 'run_gen_tag', 'run_select') as $input) {
    if (!isset($_GET[$input])) {
      $error = $error."<h1>ERROR:  Missing input parameter '$input'.</h1>";
    }
  }
  
  if ($_GET['run_select'] == 'run_range' &&
      (!isset($_GET['min_run']) || !isset($_GET['max_run']))) {
    $error = $error."<h1>ERROR:  Must provide 'min_run' and 'max_run' if using 'run_range'</h1>";
  }

  if ($_GET['run_select'] == 'date_range') {
    if ( !$_GET['min_start'] || !$_GET['max_start'] ) {
      $error = $error."<h1>ERROR:  Must provide 'min_start' and 'max_start' if using 'date_range'</h1>";
    }
    foreach (array($_GET['min_start'], $_GET['max_start']) as $date) {
      if (! strptime($date, '%Y-%m-%d %H:%M:%S') ) {
	$error = $error."<h1>ERROR:  Date '$date' is invalid.  Use the format YYYY-MM-DD hh:mm:ss.</h1>";
      }
    }
  }

  return $error;
}

function draw_data_table($datatype, $run, $run_iov_id, $runtype) {
  echo "<table class='$datatype'>";
  if     ($datatype == 'MON') { fill_monitoring_table($run, $run_iov_id, $runtype); }
  elseif ($datatype == 'DCU') { fill_dcu_table($run, $run_iov_id); }
  elseif ($datatype == 'BEAM') { fill_beam_table($run); }
  elseif ($datatype == 'LMF') { fill_laser_table($run, $run_iov_id, $runtype); }
  else { 
    echo "<tr><td class='noresults'>Data type $datatype is not finished</td></tr>";
  }
  echo "</table>";
}

function fill_monitoring_table($run, $run_iov_id, $runtype) {
  $monresults = fetch_mon_data($run_iov_id);
  $nmonrows = count($monresults['SUBRUN_NUM']);
  

  if ($nmonrows > 0) {
    
    $monselect_headers = get_monselect_headers();
    echo "<tr>";
    echo "<th class='typehead' rowspan='", $nmonrows+1, "'>MON</th>";
    foreach($monselect_headers as $db_handle => $head) {
      echo "<th>$head</th>";
    }
    echo "<th><nobr>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tasks&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</nobr></th>";
    echo "<th>DQM Pages</th>";
    echo "<th>DB Plot</th>";
    echo "</tr>";

    for ($i = 0; $i < $nmonrows; $i++) {
      echo "<tr>\n";
      foreach ($monselect_headers as $db_handle => $head) {
	echo "<td>", $monresults[$db_handle][$i], "</td>\n";
      }
      $exists_str = $monresults['DAT_EXISTS'][$i];
      $iov_id = $monresults['IOV_ID'][$i];
      $loc = $_GET['location']; // XXX function argument?
      $dqm_url = htmlentities(get_dqm_url($loc, $runtype, $run));
      $dqm_url2 = htmlentities(get_dqm_url2($loc, $runtype, $run));
      $list_bits = $monresults['TASK_LIST'][$i];
      $outcome_bits = $monresults['TASK_OUTCOME'][$i];
      echo "<td>", draw_tasklist($list_bits, $outcome_bits, $run, $loc, $iov_id), "</td>";
      echo "<td><a href='$dqm_url'>DQM</a> - <a href='$dqm_url2'>log</a></td>";
      echo "<td class='dbplot'>",  draw_plotlink('MON', $exists_str, $run, $loc, $iov_id), "</td>";
      echo "</tr>\n";
    }
  } else {
    echo "<tr>
          <th class='typehead'>MON</th>
          <td class='noresults'>No monitoring results</td></tr>";
  }
}
function fill_laser_table($run, $run_iov_id, $runtype) {
  $monresults = fetch_las_data($run_iov_id);
  $nmonrows = count($monresults['SUBRUN_NUM']);
  

  if ($nmonrows > 0) {
    
    $monselect_headers = get_lmfselect_headers();
    echo "<tr>";
    echo "<th class='typehead' rowspan='", $nmonrows+1, "'>LMF</th>";
    foreach($monselect_headers as $db_handle => $head) {
      echo "<th>$head</th>";
    }
    echo "<th>DB Plot</th>";
    echo "</tr>";

    for ($i = 0; $i < $nmonrows; $i++) {
      echo "<tr>\n";
      foreach ($monselect_headers as $db_handle => $head) {
	echo "<td>", $monresults[$db_handle][$i], "</td>\n";
      }
      $exists_str = $monresults['DAT_EXISTS'][$i];
      $iov_id = $monresults['IOV_ID'][$i];
      $loc = $_GET['location']; // XXX function argument?
      echo "<td class='dbplot'>",  draw_plotlink('LMF', $exists_str, $run, $loc, $iov_id), "</td>";
      echo "</tr>\n";
    }
  } else {
    echo "<tr>
          <th class='typehead'>LMF</th>
          <td class='noresults'>No Laser results</td></tr>";
  }
}

function fill_dcu_table($run, $run_iov_id) {
  $dcuresults = fetch_dcu_data($run_iov_id);
  $ndcurows = count($dcuresults['SINCE']);

  if ($ndcurows > 0) {
    $dcuselect_headers = get_dcuselect_headers();
    echo "<tr>";
    echo "<th class='typehead' rowspan='", $ndcurows+1, "'>DCU</th>";
    foreach($dcuselect_headers as $db_handle => $head) {
      echo "<th>$head</th>";
    }
    echo "<th>DB Plot</th>";
    echo "</tr>";
    for ($i = 0; $i < $ndcurows; $i++) {
      echo "<tr>\n";
      foreach ($dcuselect_headers as $db_handle => $head) {
	$head = $dcuresults[$db_handle][$i];
	echo "<td>", $head, "</td>\n";
      }
      $exists_str = $dcuresults['DAT_EXISTS'][$i];
      $iov_id = $dcuresults['IOV_ID'][$i];
      $loc = $_GET['location']; // XXX function argument?
      echo "<td class='dbplot'>",  draw_plotlink('DCU', $exists_str, $run, $loc, $iov_id), "</td>";
      echo "</tr>\n";
    }
  } else {
    echo "<tr>
          <th class='typehead'>DCU</th>
          <td class='noresults'>No DCU results</td></tr>";
  }
}

function fill_beam_table($run) {
  $loc = $_GET['location']; // XXX function argument?
  $beamresults = fetch_beam_data($run, $loc);
  if ($beamresults) {
    $nbeamrows = count($beamresults['RUN_NUM']);
  } else {
    $nbeamrows = 0;
  }
  
  if ($nbeamrows > 0) {
    $beamselect_headers = get_beamselect_headers();
    echo "<tr>";
    echo "<th class='typehead' rowspan='", $nbeamrows+1, "'>BEAM</th>";
    foreach($beamselect_headers as $db_handle => $head) {
      echo "<th>$head</th>";
    }

    echo "<th>Beam line data</th>";

    echo "</tr>";
    for ($i = 0; $i < $nbeamrows; $i++) {
      echo "<tr>\n";
      foreach ($beamselect_headers as $db_handle => $head) {
	$head = $beamresults[$db_handle][$i];
	echo "<td>", $head, "</td>\n";
      }

      // Uncomment for popup
      //    echo "<td class='dbplot'>",  draw_beamlink($run, $loc), "</td>";
      echo "<td class='dbplot'><a href=beam.php?run_num=$run&loc=$loc>Data</td>";
      
      echo "</tr>\n";
    }
  } else {
    echo "<tr>
          <th class='typehead'>BEAM</th>
          <td class='noresults'>No BEAM results</td></tr>";
  }
}
function draw_beamlink( $run, $loc) {
 
    $url = htmlentities("beam.php?run_num=$run&loc=$loc");
    $target = "beam$run$loc";


    echo "<div class='ttp bc'>";
     echo "<a onclick=\"return popup(this, '$target', 700)\" href='$url' >Data</a>";
    
    //   echo "<div class='rtt tt'><b>Data Available:</b><br/>";
    //  foreach (split(',', $exists_str) as $t) {
    //  echo "$t<br />";
    //  }
     echo "</div>";
     //  </div>";
 
}

function draw_plotlink($datatype, $exists_str, $run, $loc, $iov_id) {
  if ($exists_str) {
    $url = htmlentities("plot.php?run=$run&loc=$loc&datatype=$datatype&iov_id=$iov_id&exists_str=$exists_str");
    $target = "plot$datatype$iov_id";


    echo "<div class='ttp bc'>";
    echo "<a onclick=\"return popup(this, '$target', 700)\" href='$url' >Plot</a>";
    echo "<div class='rtt tt'><b>Data Available:</b><br/>";
    foreach (split(',', $exists_str) as $t) {
      echo "$t<br />";
    }
    echo "</div></div>";
  } else {
    echo "No Data Available";
  }
}

$meta = "";
if (isset($_GET['run_select']) && $_GET['run_select'] == 'last_100') {
  $meta ="<meta http-equiv='Refresh' content='300' />
          <meta http-equiv='Pragma' content='no-cache' />
          <meta http-equiv='Cache-Control' content='no-cache' />";
}

function draw_tasklist($list_bits, $outcome_bits, $run, $loc, $iov_id) {

  $tasks = get_task_array();
  $outcome = get_task_outcome($list_bits, $outcome_bits);
  
  if (! $outcome) { return; }

  foreach ($outcome as $taskcode => $result) {
    $status = $result ? 'good' : 'bad';
    $url = htmlentities("bad_channels.php?run=$run&loc=$loc&iov_id=$iov_id&taskcode=$taskcode");
    $target = "bad_channels$run$loc$iov_id$taskcode";

    echo "<div class='ttp bl $status'>
           <a onclick=\"return popup(this, '$target', 1000)\" href='$url'>$taskcode</a>
           <div class='tt' style='width:  150px'>$tasks[$taskcode]</div></div>";
  }
}

?>
<!DOCTYPE html
PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html>
<head>
<title>List Runs</title>
<?php echo get_stylelinks( $compact ); ?>
<script type="text/javascript">
<!--
function popup(mylink, windowname, width)
{
  if (! window.focus)return true;
  var href;
  if (typeof(mylink) == 'string')
  href=mylink;
  else
  href=mylink.href;
  var param  = 'width='+width+',height=600,scrollbars=yes';
  window.open(href, windowname, param);
  return false;
}
//-->
</script>
<?php echo $meta ?>
</head>
<body>

<?php
if ($errors = input_errors()) {
  echo $errors;
} else {
  $rows_per_page = 100;
  $url = htmlentities($_SERVER['REQUEST_URI']);
  $sqlresult = build_runselect_sql($_GET);
  $sql = $sqlresult['sql'];
  $binds = $sqlresult['binds'];
  
  $total_rows = count_rows($conn, $sql, $binds);
  $sum_events = fetch_sum_events($sqlresult);

  if ($total_rows == 0) {
    echo "<h3>No results found.</h3>";
  } else {
    $total_pages = total_pages($total_rows, $rows_per_page);

    // Set the page number
    if (isset($_GET['page'])) { $page = $_GET['page']; }
    if ( !isset($_GET['page']) ||
	 !preg_match('/^[0-9]+$/', $_GET['page']) ||
	 $_GET['page'] < 1 ) {
      $page = 1;
    } else if ( $_GET['page'] > $total_pages ) {
      $page = $total_pages;
    }

    $start_row = page_to_row($page, $rows_per_page);  
    $stmt = & paged_result($conn, $sql, $binds, $start_row, $rows_per_page);
    
    $runselect_headers = get_runselect_headers();
    $nruncols = count($runselect_headers);

    $datatypes = get_datatype_array();
    $ndisplay = 0;
    foreach ($datatypes as $name => $prefix) {
      if (isset($_GET[$prefix])) { $ndisplay++; }
    }


    // Start drawing the page
    // Top header and plot select
    echo "<table><tr><td>";
    // Result information
    echo "<h3>$total_rows runs returned - $sum_events events</h3>";
    echo "<h6>Showing page $page of $total_pages, $rows_per_page runs per page</h6>";
    echo "</td><td>";
    // Help information?
    foreach ($datatypes as $name => $prefix) {
      if (isset($_GET[$prefix])) {
      }
    }
    echo "</td></tr></table>";

    draw_pager($url, $total_pages, $page, "index.php"); 

    // Run table
    echo "<table class='runs'>";
    if ($ndisplay == 0) {
      echo "<tr>";
      foreach ($runselect_headers as $db_handle => $head) {
	echo "<th>$head</th>";
      }
      echo "</tr>";
    }
	   
    while ($run = oci_fetch_assoc($stmt)) {
      // Run Header
      if ($ndisplay > 0) {
	echo "<tr>";
	foreach ($runselect_headers as $db_handle => $head) {
	  echo "<th>$head</th>";
	}
	echo "</tr>";
      }
      
      //  Run data rows
      echo "<tr>";
      foreach ($runselect_headers as $db_handle => $head) {
	if ($db_handle == 'RUN_NUM') {
	  echo "<td class='run_num' rowspan='", $ndisplay+1, "'>";
	} else { echo "<td>"; }
	echo $run[$db_handle], "</td>";
      }
      echo "</tr>";
      //  Draw optional data
      foreach ($datatypes as $name => $prefix) {
	if (isset($_GET[$prefix])) {
	  echo "<tr><td colspan='",$nruncols-1, "'>";
	  draw_data_table($prefix, $run['RUN_NUM'], $run['RUN_IOV_ID'], $run['RUN_TYPE']);
	  echo "</td></tr>";
	}
      }
    }
    echo "</table>";
    
    draw_pager($url, $total_pages, $page, "index.php"); 
  }
}
?>

<pre>
<?php 


// echo "The input is:\n";
// echo var_dump($_GET);
// echo "The URL is $url\n";
// echo "The SQL is:\n$sql\n";
// echo "The binds are:\n";
// echo var_dump($binds);

?>
</pre>

</body>
</html>
