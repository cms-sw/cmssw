<?php
/*
 * pager_functions.php
 *
 * Set of functions to return a paged result from an SQL query
 * Credit:  Inspired by Harry Fuecks article
 * http://www.oracle.com/technology/pub/articles/oracle_php_cookbook/fuecks_paged.html
 *
 * $Id: pager_functions.php,v 1.3 2006/07/23 16:47:58 egeland Exp $
 */

include('Net/URL.php');

/* Returns an <a> tag with a link to $url and the page property set to $page */
function page_a($url, $page, $text) {
  $url_obj = new Net_URL($url);
  $url_obj->addQueryString('page', $page);
  $newurl = $url_obj->getURL();
  return "<a href='$newurl'>$text</a>";
}

/* Draws a pager with links to $url, given the $total_pages and $current_page*/
function draw_pager($url, $total_pages, $current_page = 1, $new_search = NULL) {
  echo "<div class='pager'>";

  if ($current_page <=0 || $current_page > $total_pages ) {
    $current_page = 1;
  }

  if ($current_page > 1) {
    echo page_a($url, 1, '[Start]'), " \n";
    echo page_a($url, ($current_page-1), '[Prev]'), " \n";
  }

  for ($i = ($current_page-5); $i <= $current_page+5; $i++) {
    if ($i < 1) continue;
    if ($i > $total_pages) break;
    
    if ($i != $current_page) {
      echo page_a($url, $i, $i), " \n";
    } else {
      echo page_a($url, $i, "<strong>$i</strong>"), " \n";
    }
  }

  if ($current_page < $total_pages) {
    echo page_a($url, ($current_page+1), '[Next]'), " \n";
    echo page_a($url, $total_pages, '[End]'), " \n";
  }

  if ($new_search) {
    echo "<a href='$new_search'>[New Search]</a>";
  }

  echo "</div>";
}

/* Returns the total number of pages given $total_rows and $rows_per_page */
function total_pages($total_rows, $rows_per_page) {
  if ($total_rows < 1) $total_rows = 1;
  return ceil($total_rows/$rows_per_page);
}

/* Returns the starting row for $current_page given $rows_per_page */
function page_to_row($current_page, $rows_per_page) {
  $start_row = ($current_page-1) * $rows_per_page +1;
  return $start_row;
}

/* Counts the number of rows given by $select.  Binding done with $binds array */
function count_rows(& $conn, $select, $binds) {
  $sql = "SELECT COUNT(*) AS num_rows FROM($select)";
  $stmt = oci_parse($conn, $sql);
  foreach ($binds as $handle => $var) {
    oci_bind_by_name($stmt, $handle, $binds[$handle]);
  } 
  oci_define_by_name($stmt, "NUM_ROWS", $num_rows);
  oci_execute($stmt);
  oci_fetch($stmt);
  return $num_rows;
}

/* Returns an statement handle containing the results of the query $select, from $start_row to ($start_row+$rows_per_page) */
function & paged_result(& $conn, $select, $binds, $start_row, $rows_per_page) {
  $end_row = $start_row + $rows_per_page -1;

  $sql = "SELECT * FROM ( SELECT r.*, ROWNUM AS row_number FROM ( $select ) r WHERE ROWNUM <= :end_row ) WHERE :start_row <= row_number";
  
  $stmt = oci_parse($conn, $sql);
  $binds[':start_row'] = $start_row;
  $binds[':end_row'] = $end_row;
  foreach ($binds as $handle => $var) {
    oci_bind_by_name($stmt, $handle, $binds[$handle]);
  } 

  oci_execute($stmt);

  oci_set_prefetch($stmt, $rows_per_page);
  
  return $stmt;
}
?>
