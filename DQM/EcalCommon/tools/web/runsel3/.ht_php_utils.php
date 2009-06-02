<?php

  function tr_alt($i=-1) {
    global $ialt;
    if ($i != -1) { $ialt = $i; }
    if ($ialt == "") { $ialt = 0; }
    if ($ialt == 0) { echo "<tr>"; $ialt = 1; }
      else { echo "<tr class=\"light\">"; $ialt = 0; }
  }

  function tr_alt2($i=-1) {
    global $ialt;
    if ($i != -1) { $ialt = $i; }
    if ($ialt == "") { $ialt = 0; }
    if ($ialt == 0) { $ialt = 1; return "<tr>"; }
      else {$ialt = 0;  return "<tr class=\"light\">"; }
  }


  function equal_zero($var) {
     if ($var == "0") {
         return true;
     } else {
         return false;
     }
  }


#  read recursively the table meta_tablelinks for tables listed in tblin keys
  function create_join(&$tblin) {
    $listtbl = implode(",", array_keys(array_filter($tblin, "equal_zero")));
    $req_sql = "select meta_tablelinks.tablename,meta_tablelinks.joinstr,
                       meta_tablelinks.tableneeded
                from meta_tablelinks
                where (meta_tablelinks.tablename in (".$listtbl."))
                  and (meta_tablelinks.listby = 'run')
                group by meta_tablelinks.tablename";

    ldb_sql($req_sql);
    $joinstr = "";
    $fgneeded = FALSE;
    while ( ldb_fetch() ) {
      $table = "'".ldb_result(1)."'";
      $join = ldb_result(2);
      $needed = ldb_result(3);
      if ($join != "") $joinstr .= "left join ".$join." ";
      $tblin[$table] = $join;
      if (($needed != "") && (!array_key_exists("'".$needed."'",$tblin))) {
        $fgneeded = TRUE;
        $tblin["'".$needed."'"] = 0;
      }
    }
    ldb_endsql();
    $joinneeded = "";
    if ($fgneeded) $joinneeded = create_join($tblin);
    $joinneeded .= $joinstr;
    return $joinneeded;
  }


# to print float in scientific notation
  function sciprint($x, $d=3) {
    $min=($x<0)?"-":"";
     $x=abs($x);  
    $e=floor(($x!=0)?log10($x):0);
     $x*=pow(10,-$e);
    $fmt=($d>=0)?".".$d:"";
    $e=($e>=0)?"+".sprintf("%02d",$e):"-".sprintf("%02d",-$e);
    return sprintf("$min%".$fmt."fe%s",$x,$e);
  }

# to print date&time from unix time
  function timeprint($val) {
    return Date('j/m H:i',$val);
  }

# to print date&time from unix time
  function hourprint($val) {
    return Date('H:i:s',$val);
  }


?>
