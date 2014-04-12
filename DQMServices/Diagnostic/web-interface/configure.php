<?php
$GLOBALS['userName'] = null;
if(!isset($GLOBALS['userName']))
  $GLOBALS['userName'] = substr($_SERVER['PHP_SELF'], 1, (strpos($_SERVER['PHP_SELF'],'/',1)-1) );
//$GLOBALS['backendAddress'] = "webcondvm.cern.ch";
//$GLOBALS['backendPort'] = "8083";
?>
<script>
  var userNameJs = "<?=$GLOBALS['userName']?>";
</script>
