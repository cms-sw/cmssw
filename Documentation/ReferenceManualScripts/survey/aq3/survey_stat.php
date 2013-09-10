<?php

try
  { 
  	$db = new PDO('sqlite:aq.db');

	$question_count = $db->query("SELECT COUNT(*) FROM Question WHERE is_separator = 0")->fetch();

	$questionnaires = $db->query("SELECT * FROM Questionnaire ORDER BY publication_number ASC")->fetchAll();

	$pags = array();

	foreach($questionnaires as $questionnaire){

		$publication = $questionnaire['publication_number'];

		$items = explode("-", $publication);

		if (sizeof($items) > 1){
			$pag = $items[0];
			if ($pag == "CMS"){
				$pag = $items[1];
			}

			if (!array_key_exists($pag, $pags)){
				$pags[$pag] = array();
			}

	$questions_answered = $db->query("SELECT count(distinct(q.id)) FROM Response r
			JOIN Option AS o ON r.option_id = o.id
			JOIN Question AS q ON o.question_id = q.id
			WHERE 
			(r.text<> '' OR r.selected = 1) AND 
			q.is_separator = 0 AND
			r.questionnaire_id = ".$questionnaire['id'])->fetch();

			$pags[$pag][] = " (".$questions_answered[0].") <a target='_blank' href='http://cmsdoxy.web.cern.ch/cmsdoxy/aq3/questionnaire.php?id=".$questionnaire['id']."'>".$publication."</a>";			
		}	
	}

	function cmp($a, $b) {
        	return sizeof($a) - sizeof($b);
	}
	uasort($pags, "cmp");



	$total_publications = 0;

	foreach($pags as $pag => $publications){
		echo "<div style='float:left; width:180px; padding:10px; margin:5px; text-align: centere; border: 1px dashed black;'>";
		echo "<b>".$pag." (".sizeof($publications).")</b>";
		echo "<br/><br/>";
		foreach ($publications as $publication){
		    echo $publication . "<br>";
		}
		echo "</div>";
		$total_publications += sizeof($publications);
	}

	echo "<div style='float:left; margin:5px; text-align: center'>";
	echo "<b>Total responses so far</b><br/>".$total_publications;
	echo "</div>";

}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>

<div style="clear:both"></div>
<hr>
<b>(</b>number of answered questions<b>) PUBLICATION_NUMBER </b>
<br/>
number of total questions: <b><?php echo $question_count[0]; ?></b>
