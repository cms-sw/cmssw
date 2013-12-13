#!/bin/bash

# check parameters
if [[ "$#" != "2" ]] ; then
  echo "Usage: ./cmpr.sh test.dat [html|text]" >&2
  exit 1
fi

TEST_DAT="$1"
FORMAT="$2"
REFERENCE_DAT="reference.dat"

if ! [ -f ${TEST_DAT} ] || ! [ -s ${TEST_DAT} ] ; then
  echo "ERROR: $0: ${TEST_DAT} doesn't exist or empty." >&2
  exit 1
fi

#----------------------------

top_text () {
  printf "%-15s %5s | %-12s %-12s   %-12s %-12s | %10s %-10s\n" "Routine" "Test"  "Y" "dY"  "Yref" "dYref"  "Pull" "Status"
  echo "-------------------------------------------------------------------------------------------------"
}

top_html () {
cat << EOT
<html>
<head>
  <title>Test results</title>
  <style>
    tr#good {}
    .switchlink {color: #0055CC; cursor: pointer; margin: 4px; border-bottom: 1px dashed #0055CC; }
  </style>
  
  <script language="javascript">
<!--
function turn_good()
{
  var rows = document.getElementsByTagName('tr');
  for (var i = 0; i < rows.length; i++)
  {
    var row = rows[i];

    if (row.id == "good") {
      if (row.style.display != "none")
        row.style.display = "none";
      else
        row.style.display = "table-row";
    }
  }
}
-->
  </script>
</head>
<body>
<h1>Test results</h1>
<p>
Platform: ${PLATFORM}<br>
Test date: $(date)<br>
<hr>
<h4>Notation:</h4>
<table cellpadding=3>
<tr>
  <td align=left> Y , &nbsp; &nbsp; &nbsp; dY </td>
  <td> -- value of an observable and its stat. error  </td>
</tr>
<tr>
  <td align=left> Y<sub>ref</sub> ,  dY<sub>ref</ref> </td> 
  <td> -- reference value of an observable and its stat. error  </td>
</tr>
<tr>
  <td align=left> Pull  </td> 
  <td> -- ( Y - Y<sub>ref</sub> ) / ( dY<sup> 2</sup> + dY<sup> 2</sup><sub>ref</sub> )<sup> 1/2</sup> </td>
</tr>
<tr>
  <td bgcolor="#00ff00" align=center>ok</td> 
  <td> -- test is succesfully compiled and executed with <i>pull</i> &lt; 3 </td>
</tr>
<tr>
  <td bgcolor="#4aa1ff" align=center>badstat</td>
  <td> -- as above, but statistics is insufficient: Y<sub>ref</sub> &lt; 5dY<sub>ref</sub> or Y &lt; 4dY </td>
</tr>
<tr>
  <td bgcolor="#ffaa00" align=center> deviation</td> 
  <td> -- <i>pull</i> &gt; 3 </td>
</tr>
<tr>
  <td bgcolor="#ff0000" align=center> failed </td>
  <td> -- test crashed </td>
</tr>
<tr>
  <td bgcolor="#ff0000" align=center> errors </td>
  <td> -- test failed to compile </td>
</tr>
</table>
<hr>

<p>
  <span class="switchlink" onClick="turn_good()">Show/Hide rows with Status = [OK]</span>
</p>

<table border=1>
<tr>
  <td>Generator</td>
  <td>Test</td>
  <td>Y</td>
  <td>dY</td>
  <td>Y<sub>ref</sub></td>
  <td>dY<sub>ref</sub></td>
  <td>Pull</td>
  <td>Status</td>
</tr>

EOT
}

bottom_html () {
  echo "</table>";
  return;
}

print_text () {
  printf "%-15s %5s | %-12s %-12s   %-12s %-12s | %10s %-10s\n" $*
}

print_html () {
  local g=$(echo $* | cut -d ' ' -f 1)
  local n=$(echo $* | cut -d ' ' -f 2)
  local end=$(echo $* | cut -d ' ' -f 3-)
  
  local rowid="bad"
  if [ "x$end" != "x${end/OK/}" ] ; then
    rowid="good"
  fi
  
  local color
  if   [[ "$end" != "${end/OK/}" ]] ; then
    color="#00ff00"
  elif [[ "$end" != "${end/NO_REFERENCE/}" ]] ; then
    color="#FFD91C"
  elif [[ "$end" != "${end/BADSTAT/}" ]] ; then
    color="#4aa1ff"
  elif [[ "$end" != "${end/DEVIATION/}" ]] ; then
    color="#ffaa00"
  else
    color="#ff0000"
  fi
  
  echo "<tr id=\"$rowid\">"
  echo "  <td> <a href=\"#${g}\"> ${g} </a> </td>"
  echo "  <td align=right> $n </td>"
  
  printf "<td>%s</td> <td>%s</td>   <td>%s</td> <td>%s</td>  <td align=right>%s</td> <td align=center bgcolor=\"$color\">%s</td> </tr>\n" $end
}

process_tests () {
  cc -o chi.exe chi.c -lm >&2
  
  # TODO: check the case of empty $TEST_DAT file = [GENERATOR FAILED!]
  
  {
    # print the list of tested generators
    # [output]:
    #   "generator name"
    
    cat $TEST_DAT | cut -d _ -f 1 | sort | uniq
  } | \
  {
    # print the list of all available tests for
    # generators from the input stream
    # [output]:
    #   "test name" "test #"
    
    while read gen ; do
      cat $TEST_DAT $REFERENCE_DAT | sed 's,!.*$,,' | grep "^${gen}_" | sed 's,  *, ,g' | cut -d ' ' -f 1-2 | sort -u
    done
  } | \
  {
    # print the list of test results for
    # the tests from the input stream
    # [output]:
    #   "test name" "test #" "y" "dy" "y0" "dy0" "chi2" "result"
    
    while read gentest no ; do
      
      output="$gentest $no "

      # get test and reference values
#      echo "DBG:^$gentest $no:" >&2
      ydy=$(cat $TEST_DAT | sed 's,  *, ,g; s,!.*$,,;' | grep -E "^${gentest} ${no} " | cut -d ' ' -f 3-4)
#      echo "DBG ydy:$ydy:">&2
      y0dy0=$(cat $REFERENCE_DAT | sed 's,  *, ,g; s,!.*$,,;' | grep -E "^${gentest} ${no} " | cut -d ' ' -f 3-4)
#      echo "DBG y0dy0:$y0dy0:">&2
      
      if [[ "$ydy" != "" && "$y0dy0" != "" ]] ; then
#	      echo "DBG: --- echo $ydy $y0dy0 | chi --- " >&2
        output1=`echo "$ydy $y0dy0" | ./chi.exe`
	echo "$output $output1"
      else
        if [[ "$y0dy0" == "" ]] ; then
          # reference is missing
          y0dy0="- -"
          result="[NO_REFERENCE]"
        fi
        
        if [[ "$ydy" == "" ]] ; then
          # test is missing
          ydy="- -"
          result="[TEST_FAILED]"
        fi
        
        # NOTE: if both - reference and test is missing
        #       then the test will be marked as FAILED
        
        echo "$output $ydy $y0dy0 - $result"
      fi
    done
  }
  
  rm -f chi.exe >&2
}

# print test dependencies
print_tests_info() {
  # find library dependencies tool
  if which ldd >& /dev/null ; then
    # linux machine:
    LDD="ldd "
  elif which otool >& /dev/null ; then
    # mac machine:
    LDD="otool -L "
  else
    echo "WARNING: can't find ldd routine. Can't get dynamically linked libraries the tests are compiled against." >&2
    LDD="true "
  fi
  
}

#---------------------

if [[ "${FORMAT}" == "text" ]] ; then
  top_text
  process_tests | while read line ; do print_text $line ; done
else
  top_html
  process_tests | while read line ; do print_html $line ; done
  bottom_html
  print_tests_info
fi
