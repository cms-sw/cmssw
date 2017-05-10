<html>
<body>

<?php

function putURLs($sender, $URLs){

//	$lines = file('encoded_users.txt');
//        $found = false;
//
//	foreach ($lines as $line) {
//		$line_elements = explode(" ", $line);
//  
//		if ($sender == $line_elements[0]){
//			$found = true;
//			$sender = $line_elements[1];
//		}
//	}
//
//	if (!$found){
//		die("Error: can't find such user. <b> PLEASE IMEDIATELY contact mantas.stankevicius@cern.ch </b>");
//	}

 	$file = fopen("responses.txt", 'a+')  or die("Error: can't process your request. <b> PLEASE IMEDIATELY contact mantas.stankevicius@cern.ch </b>");

	fwrite($file, "--------".date('l jS \of F Y h:i:s A')."\n");
	fwrite($file, "Sender: ".$sender."\n\n");
	fwrite($file, $URLs."\n");
	fwrite($file, "------------------------------------------------\n\n");	
	fclose($file);

	echo("<p><b>Deletion request for the following twiki pages has been successfully sent!</b></p>");
	echo("<pre>".$URLs."</pre>");


}

$URLS = array();
$BASE = "https://twiki.cern.ch/twiki/bin/viewauth/";
$SENDER = "";

	$i = 0;
	if (count($_POST) > 0) $list = $_POST;
	else $list = $_GET;
	foreach($list as $k => $v) {
        	if ($k == "sender") continue;
        	$URLS[$i] = $BASE.$k;
        	$i++;
	}
	$U = array_unique($URLS);
        $SENDER = $list['sender'];

	if (count($U) > 0) {
		$CONTENT = implode("\n", $U);
		putURLs($SENDER, $CONTENT);
	}
	else {
		echo("<p><b>You didn't select any pages for deletion request!</b></p>");
		}
	echo("<div>For further questions: mantas.stankevicius@cern.ch</div>"); 
	echo("<div><input type='submit' value='Go Back' onclick='javascript: history.back()' /></div>");

?>
</body>
<html>
