<body style="font-family:arial,sans-serif">
<link href="http://twitter.github.com/bootstrap/assets/css/bootstrap.css" rel="stylesheet">

<?php
function safe($str){

	$str = str_replace('"', "&#34", $str);
	$str = str_replace("'", "&#39", $str);
	$str = str_replace("<", "&lt;", $str);
	$str = str_replace(">", "&gt;", $str);

	return $str;
}

function printSafe($str){
	return stripslashes($str);
}

function printOption($option, $response){	

$ret = "";

	if ($option['is_checkbox'] == 1 && $response['selected'] == 1){			
		$ret.= "<b>".printSafe($option['title'])."</b><br/>";
	}
	elseif ($option['is_radio'] == 1 && $response['selected'] == 1){
		$ret.= "<b>".printSafe($option['title'])."</b><br/>";
	}	
	elseif ($option['is_text']==1) {
		$ret.= "".printSafe($option['title']). " : ";
		$ret.= "<b>".printSafe($response['text'])."</b><br/>";
	}
	elseif ($option['is_comment']==1){
		$ret.= printSafe($option['title'])." : ";
	}
	
	
	if ( ( ($option['is_radio'] == 1) || ($option['is_checkbox'] == 1) ) && ($option['is_commentable']==1) ){
		$ret.= "<b>".printSafe($response['text'])."</b><br/>";
	}
	
	
	return $ret;
}

try{

	$db = new PDO('sqlite:aq.db');

	$qid = safe($_GET['questionnaire']);
	
	$questionnaires = $db->query('SELECT * FROM Questionnaire ORDER BY publication_number ASC')->fetchAll();


	echo "<div style='float:left; width:200px'>";
        echo "<center><h3>Surveys</h3></center>";
	echo "<table class='table table-hover'>";
	foreach($questionnaires as $questionnaire){
		
		echo "<tr><td><a href='?questionnaire=".$questionnaire['id']."'>";
		if ($qid == $questionnaire['id']){
			echo "<b>".$questionnaire['publication_number']."</b>";
		}
		else{
			echo $questionnaire['publication_number'];
		}
		echo "</a></td></tr>";
	
	}

echo "</table></div><div style='float:left; width:700px; margin-left:50px'>";

$qid = safe($_GET['questionnaire']);
if ($qid != ""){


	$questions = $db->query('SELECT * FROM Question ORDER BY place ASC')->fetchAll();
	
	foreach($questions as $question){	
		 if ($question['is_separator']==1){
			echo "<h3>".$question['title']."</h3>";
		 }

	  $options = $db->query('SELECT * FROM Option WHERE question_id='.$question['id'].' ORDER BY place ASC')->fetchAll();	

	  $html = "";
	  $empty = false;

	  foreach($options as $option){			

		$response = $db->query("SELECT * FROM Response WHERE option_id='".$option['id']."' AND questionnaire_id='".$qid."'")->fetch();

		if (($option['title'] == "Name") && ($response['text'] == "")){
			$empty = true;
			break;
		}

		$html .= printOption($option, $response);               

	  }

	 if ($empty == false){
		if ($question['is_separator']==0){		
			echo "<u>".$question['title']."</u><br/>";
		 }
	        
		 echo $html;

		 echo "<br/>"; // question 
	  }

	  
    	}
}
else{
	echo "<center><br/><br/><br/><br/><br/><br/><h3>Select survey from list </h3></center>";
}


echo "</div><div style='clear:both></div>";

}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }

?>

</body>